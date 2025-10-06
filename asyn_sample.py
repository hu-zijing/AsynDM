import contextlib
import os
import datetime
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from diffusion.asyn_ddim_with_logprob import ddim_step_with_logprob, asyn_ddim_step_with_logprob, latents_decode
from diffusers import DDIMScheduler
from model.unet_2d_condition import unet_asyn_forward
import torch
import torch.nn.functional as F
from functools import partial
import tqdm
from PIL import Image
import json
import random
from utils.utils import seed_everything

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Sampling configuration.")

logger = get_logger(__name__)

def main(_):
    # basic setup
    config = FLAGS.config
    debug_idx = 0
    print(f'========== seed: {config.seed} ==========')
    torch.cuda.set_device(config.dev_id)

    unique_id = config.exp_name if config.exp_name else datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    save_dir = os.path.join(config.save_path, unique_id)

    seed_everything(config.seed)

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir, 
        automatic_checkpoint_naming=True,
        total_limit=100,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )

    # load
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) # float16
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    autocast = accelerator.autocast

    prompt_list = [config.prompt]
    if len(config.prompt_file)!=0:
        with open(config.prompt_file, 'r') as f:
            prompt_list = json.load(f)
    # print('prompt list:', prompt_list)
    prompt_cnt = len(prompt_list)
    total_num_batches_per_epoch = config.sample.num_batches_per_epoch*prompt_cnt

    item_idx_list = [config.item_idx]*prompt_cnt
    item_k_list = [config.item_k]*prompt_cnt
    if len(config.item_idx_file)!=0:
        with open(config.item_idx_file, 'r') as f:
            temp_list = json.load(f)
            item_idx_list = [temp_list["item_idx"][p] for p in prompt_list]
            item_k_list = [temp_list["item_k"][p] for p in prompt_list]

    def func_prev_linear(state_t, rest_step, target_value = pipeline.scheduler.config.steps_offset):
        if rest_step == 0: 
            state_prev_t = torch.zeros_like(state_t)
        else: 
            target_value_tensor = torch.tensor(target_value, dtype=state_t.dtype, device=state_t.device)
            state_prev_t = (state_t*(1-1/rest_step) + target_value_tensor/rest_step)
        return state_prev_t

    def func_prev_binary(
        state_t, rest_step, k=0.5, 
        target_value = pipeline.scheduler.config.steps_offset,
        curve_type=config.curve_type,
        x_scaling = config.sample.num_steps, 
        y_scaling = pipeline.scheduler.config.num_train_timesteps
        ):
        if rest_step == 0: 
            state_prev_t = torch.zeros_like(state_t)
        else: 
            if curve_type == "bin": 
                decay_k = y_scaling / (x_scaling*x_scaling)
                k = -k*decay_k
                target_value_tensor = torch.tensor(target_value, dtype=state_t.dtype, device=state_t.device)
                t0 = (k * rest_step**2 - (target_value_tensor - state_t)) / (2 * k * rest_step)
                y0 = state_t - k * t0**2
                state_prev_t = (k * (1 - t0)**2 + y0)
            elif curve_type == "lin": 
                k1 = -(1-k)*y_scaling/x_scaling
                k2 = -(1+k)*y_scaling/x_scaling
                c1 = state_t
                c2 = target_value - k2*rest_step
                state_prev_t = (k1+c1).clamp(max=k2+c2)
            elif curve_type == "exp":
                # y = a*e^(lamb*x)+bx+c
                lamb = 1/x_scaling
                a = -k*y_scaling / (np.e-1)
                b = -(1-k)*y_scaling / x_scaling
                # state_prev_t = k*(np.e**lamb)+a+b
                exp_neg_lamb_x0 = (state_t-target_value+b*rest_step) / (a*(1-np.e**(lamb*rest_step))) # e^(-lamb*x_0)
                state_prev_t = a*(np.e**lamb-1) * exp_neg_lamb_x0 + b + state_t
            else: 
                raise ValueError(f"wrong curve type: {curve_type}")
        return state_prev_t

    pipeline.unet.eval()
    total_prompts1 = []
    global_idx = config.begin_index * config.sample.batch_size
    local_idx = 0
    if global_idx: 
        with open(os.path.join(save_dir, f'prompt.json'),'r') as f:
            total_prompts1 = json.load(f)[:global_idx]
    for idx in tqdm(
        range(config.begin_index, total_num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # generate prompts
        prompt_idx = idx//config.sample.num_batches_per_epoch
        prompts1 = [
            prompt_list[prompt_idx]
            for _ in range(config.sample.batch_size)
            ] 
        total_prompts1.extend(prompts1)
        # encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        # combine prompt and neg_prompt
        prompt_embeds1_combine = torch.cat([sample_neg_prompt_embeds, prompt_embeds1], dim=0)

        # ================================================================= #
        # base
        cross_mask = None

        if config.generate_base or config.static_mask: 
            gs = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs): 
                g.manual_seed(config.seed+(idx%config.sample.num_batches_per_epoch)*config.sample.batch_size+i)
            noise_latents1 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds1.dtype, 
                accelerator.device, 
                gs ## generator
            )
            
            pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
            # timestep_list = [i*pipeline.scheduler.config.num_train_timesteps//config.sample.num_steps+pipeline.scheduler.config.steps_offset for i in range(config.sample.num_steps)][::-1]
            # print(timestep_list)
            ts = pipeline.scheduler.timesteps

            extra_step_kwargs = pipeline.prepare_extra_step_kwargs(gs, config.sample.eta)

            latents_t = noise_latents1
            cross_mask = []

            for i, t in tqdm(
                enumerate(ts),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                # sample

                with autocast():
                    with torch.no_grad():
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)

                        noise_pred, extra_inf = unet_asyn_forward(pipeline.unet,
                            latents_input,
                            t,
                            # concat_t,
                            encoder_hidden_states=prompt_embeds1_combine, 
                            return_dict=False, 
                            extra_input = {
                                'used_layer_size':16,
                                'item_idx':item_idx_list[prompt_idx]
                                }, 
                            return_extra_inf = True,
                        )
                        noise_pred = noise_pred[0]
                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_t_1, _, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, i, latents_t, **extra_step_kwargs)
                        latents_t = latents_t_1

                        cross_mask.append(extra_inf['cross_mask'])

            cross_mask = torch.stack(cross_mask, dim=0).mean(dim=0)
            mask_mean = config.mask_thr * cross_mask.mean(dim=1, keepdim=True)
            cross_mask[cross_mask >= mask_mean] = 1
            cross_mask[cross_mask < mask_mean] = 0

            bsize, width_height, item_cnt = cross_mask.shape
            width = int(width_height**0.5)
            cross_mask = cross_mask.permute(0,2,1).reshape(bsize,item_cnt,width,width)
            a_tensor = torch.tensor(item_k_list[prompt_idx], dtype=torch.float32, device=cross_mask.device)  # shape: (item_cnt,)
            a_tensor = a_tensor.view(1, item_cnt, 1, 1)  # shape: (1, item_cnt, 1, 1)
            priority_masks = cross_mask * a_tensor  # (bsize, item_cnt, width, width)
            _, max_idx = priority_masks.max(dim=1)  # shape: (bsize, width, width)
            final_masks = torch.zeros_like(cross_mask)  # (bsize, item_cnt, width, width)
            for i in range(item_cnt):
                final_masks[:, i] = (max_idx == i).float() * cross_mask[:, i]
            cross_mask = final_masks
            cross_mask = F.interpolate(cross_mask, (64,64)) # mode: nearest

            if config.generate_base: 
                images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1.dtype).cpu().detach()

                os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
                for j, image in enumerate(images):
                    # print(image)
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}_base.png"))

        # ================================================================= #
        # base2

        if config.generate_base2: 
            gs = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
            for i,g in enumerate(gs): 
                g.manual_seed(config.seed+(idx%config.sample.num_batches_per_epoch)*config.sample.batch_size+i)
            noise_latents1 = pipeline.prepare_latents(
                config.sample.batch_size, 
                pipeline.unet.config.in_channels, ## channels
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
                prompt_embeds1.dtype, 
                accelerator.device, 
                gs ## generator
            )
                
            max_k = max(item_k_list[prompt_idx])
            initial_t = pipeline.scheduler.config.num_train_timesteps+pipeline.scheduler.config.steps_offset
            # print(max_k, initial_t)
            initial_t = torch.tensor(initial_t, device=accelerator.device, dtype=torch.float32)
            state_t = initial_t[None, None, None].expand(config.sample.batch_size, 64, 64)
            state_t = func_prev_binary(state_t, config.sample.num_steps, k=max_k)
            # print(state_t)

            extra_step_kwargs = pipeline.prepare_extra_step_kwargs(gs, config.sample.eta)

            latents_t = noise_latents1
            for i in tqdm(
                range(config.sample.num_steps),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                # sample

                with autocast():
                    with torch.no_grad():
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input)

                        # print(state_t)
                        concat_t = torch.cat([state_t.reshape(-1, 64*64)] * 2).round().long()
                        state_prev_t = func_prev_binary(state_t, config.sample.num_steps-i-1, k=max_k)
                        # print(state_t, state_prev_t)

                        tensor_t = state_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()
                        tensor_prev_t = state_prev_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()
                        
                        noise_pred = unet_asyn_forward(pipeline.unet,
                            latents_input,
                            # t,
                            concat_t,
                            encoder_hidden_states=prompt_embeds1_combine, 
                            return_dict=False, 
                        )
                        noise_pred = noise_pred[0]
                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents_t_1, _, latents_0 = asyn_ddim_step_with_logprob(pipeline.scheduler, noise_pred, tensor_t, tensor_prev_t, latents_t, **extra_step_kwargs)
                        latents_t = latents_t_1

                        state_t = state_prev_t

            images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1.dtype).cpu().detach()

            os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}_base2.png"))

        # ================================================================= #
        # asyn

        gs = [torch.Generator(device='cuda') for _ in range(config.sample.batch_size)]
        for i,g in enumerate(gs): 
            g.manual_seed(config.seed+(idx%config.sample.num_batches_per_epoch)*config.sample.batch_size+i)
        noise_latents1 = pipeline.prepare_latents(
            config.sample.batch_size, 
            pipeline.unet.config.in_channels, ## channels
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
            prompt_embeds1.dtype, 
            accelerator.device, 
            gs ## generator
        )
        
        item_cnt = len(item_idx_list[prompt_idx])
        if not config.static_mask: 
            cross_mask = torch.zeros(config.sample.batch_size, item_cnt, 64, 64, dtype=torch.float32, device=accelerator.device)
            cross_mask[:, np.array(item_idx_list[prompt_idx]).argmax()] = 1
        bg_mask = 1 - (cross_mask > 0.5).any(dim=1).float()
        initial_t = pipeline.scheduler.config.num_train_timesteps+pipeline.scheduler.config.steps_offset
        initial_t = torch.tensor(initial_t, device=accelerator.device, dtype=torch.float32)
        state_t = initial_t[None, None, None].expand(config.sample.batch_size, 64, 64)
        state_prev_t_linear = func_prev_linear(state_t, config.sample.num_steps)
        state_prev_t_binary = []
        for j in range(item_cnt): 
            state_prev_t_binary.append(cross_mask[:,j]*func_prev_binary(state_t, config.sample.num_steps, k=item_k_list[prompt_idx][j]))
        state_prev_t_binary = torch.stack(state_prev_t_binary, dim=1).sum(dim=1)
        state_t = (bg_mask*state_prev_t_linear + state_prev_t_binary)
        # print(state_t)

        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(gs, config.sample.eta)

        latents_t = noise_latents1
        frames = [[] for _ in range(config.sample.batch_size)]

        for i in tqdm(
            range(config.sample.num_steps),
            desc="Timestep",
            position=3,
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):  
            # sample

            with autocast():
                with torch.no_grad():
                    latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                    latents_input = pipeline.scheduler.scale_model_input(latents_input)

                    # print(state_t)
                    concat_t = torch.cat([state_t.reshape(-1, 64*64)] * 2).round().long()
                    
                    bg_mask = 1 - (cross_mask > 0.5).any(dim=1).float()
                    state_prev_t_linear = func_prev_linear(state_t, config.sample.num_steps-i-1)
                    state_prev_t_binary = []
                    for j in range(item_cnt): 
                        state_prev_t_binary.append(cross_mask[:,j]*func_prev_binary(state_t, config.sample.num_steps-i-1, k=item_k_list[prompt_idx][j]))
                    state_prev_t_binary = torch.stack(state_prev_t_binary, dim=1).sum(dim=1)
                    state_prev_t = (bg_mask*state_prev_t_linear + state_prev_t_binary)

                    tensor_t = state_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()
                    tensor_prev_t = state_prev_t[:, None].expand(config.sample.batch_size, 4, 64, 64).round().long()
                    
                    noise_pred, extra_inf = unet_asyn_forward(pipeline.unet,
                        latents_input,
                        # t,
                        concat_t,
                        encoder_hidden_states=prompt_embeds1_combine, 
                        return_dict=False, 
                        extra_input = {
                            'used_layer_size':16,
                            'item_idx':item_idx_list[prompt_idx]
                            }, 
                        return_extra_inf = True,
                    )
                    noise_pred = noise_pred[0]
                    if config.sample.cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents_t_1, _, latents_0 = asyn_ddim_step_with_logprob(pipeline.scheduler, noise_pred, tensor_t, tensor_prev_t, latents_t, **extra_step_kwargs)
                    latents_t = latents_t_1

                    if not config.static_mask: 
                        cross_mask = extra_inf['cross_mask'] # (bsize, width_height, item_idx)
                        mask_mean = config.mask_thr * cross_mask.mean(dim=1, keepdim=True)
                        cross_mask[cross_mask >= mask_mean] = 1
                        cross_mask[cross_mask < mask_mean] = 0

                        bsize, width_height, item_cnt = cross_mask.shape
                        width = int(width_height**0.5)
                        cross_mask = cross_mask.permute(0,2,1).reshape(bsize,item_cnt,width,width)
                        a_tensor = torch.tensor(item_k_list[prompt_idx], dtype=torch.float32, device=cross_mask.device)  # shape: (item_cnt,)
                        a_tensor = a_tensor.view(1, item_cnt, 1, 1)  # shape: (1, item_cnt, 1, 1)
                        priority_masks = cross_mask * a_tensor  # (bsize, item_cnt, width, width)
                        _, max_idx = priority_masks.max(dim=1)  # shape: (bsize, width, width)
                        final_masks = torch.zeros_like(cross_mask)  # (bsize, item_cnt, width, width)
                        for j in range(item_cnt):
                            final_masks[:, j] = (max_idx == j).float() * cross_mask[:, j]
                        cross_mask = final_masks
                        cross_mask = F.interpolate(cross_mask, (64,64)) # mode: nearest

                    state_t = state_prev_t

        images = latents_decode(pipeline, latents_t, accelerator.device, prompt_embeds1.dtype).cpu().detach()

        os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
        for j, image in enumerate(images):
            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}_tgt.png"))
        global_idx += len(images)
        local_idx += len(images)
        with open(os.path.join(save_dir, f'prompt.json'),'w') as f:
            json.dump(total_prompts1, f)

if __name__ == "__main__":
    app.run(main)
