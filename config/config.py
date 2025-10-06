import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # random seed for reproducibility.
    config.seed = 1234
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16" # "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # sample path
    config.save_path = "./data"
    # exp name
    config.exp_name = "" # "test"
    # gpu id
    config.dev_id = 0
    # prompt directly used
    config.prompt = "a rabbit playing basketball"
    # prompt file
    config.prompt_file = ""
    # cross mask threshold
    config.mask_thr = 1.0
    # item idx in prompt
    config.item_idx = [1,3] # [1,5,11][1,7,13][3,9][1,4][2,4]
    # item k in prompt
    config.item_k = [0.7, 0.7]
    # use static or dynamic mask
    config.static_mask = 0
    # item idx file
    config.item_idx_file = ""
    # whether generate base2
    config.generate_base2 = 1
    # whether generate base
    config.generate_base = 1
    # batch begin index
    config.begin_index = 0
    # curve type
    config.curve_type = "bin" # "bin", "lin", "exp"

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "path/to/your/sd2.1-base" # or "stabilityai/stable-diffusion-2-1"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 8
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 1
    # whether use classifier-free guidance
    sample.cfg = True

    return config
