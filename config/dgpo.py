import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))
def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # DGPO Hyper
    config.use_dpm = True
    config.flow_shift = 3
    config.sample.noise_level = 0

    config.num_train_timesteps = 4
    config.trunc_steps = 6 # equals to t_min = shift(0.3)
    config.use_shared_noise = True
    config.switch_ema_ref = 200
    config.train.beta_dpo = 100
    config.clip_range = 5e-2
    # Set clip_range to be 1e-2, 1e-3 can be more stable but slower training
    # config.clip_range = 1e-2
    # config.clip_range = 1e-3

    # PPO-style Clipping
    config.clip_dsm = True
    config.clip_kl = False
    
    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 14
    config.sample.guidance_scale = 4.5
    config.sample.noise_level = 0

    config.clip_range = 1e-2
    config.clip_dsm = True
    config.switch_ema_ref = 500

    config.train.beta = 0.02
    config.trunc_steps = 4 # equals to t_min = shift(0.3)

    config.resolution = 512
    config.sample.train_batch_size = 8 * 2
    config.sample.num_image_per_prompt = 16 # set to be 24 for better performance.
    num_groups = 8 # set to be 24 for better performance.
    config.sample.num_batches_per_epoch = int(num_groups/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch # Update once per epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/dgpo_sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }

    config.train.beta_dpo = 101
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 14
    config.sample.guidance_scale = 4.5

    config.clip_range = 1e-2
    config.resolution = 512
    config.sample.train_batch_size = 8 * 2
    config.sample.num_image_per_prompt = 16 # set to be 24 for better performance.
    num_groups = 8 # set to be 24 for better performance.
    config.sample.num_batches_per_epoch = int(num_groups/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.beta = 0.02


    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch # Update once per epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.sample.noise_level = 0
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/dgpo_sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config
    
def get_config(name):
    return globals()[name]()
