
from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import DPMSolverMultistepScheduler
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import torch.distributed as dist


metrics_history = defaultdict(list)


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def generate_shared_sampled_timesteps(accelerator, train_timesteps, M):
    """
    跨 GPU 生成共享的采样 timesteps。
    
    Args:
        accelerator: Accelerator 对象
        train_timesteps: 可采样的 timestep 列表
        M: 要采样的数量
    
    Returns:
        sampled_timesteps: 列表，所有 GPU 上的值相同
    """
    device = accelerator.device
    
    if accelerator.is_main_process:
        sampled = random.sample(train_timesteps, M)
        sampled_tensor = torch.tensor(sampled, device=device, dtype=torch.long)
    else:
        sampled_tensor = torch.empty(M, device=device, dtype=torch.long)
    
    dist.broadcast(sampled_tensor, src=0)
    
    return sampled_tensor.tolist()

def generate_shared_noise_for_groups(x0, group_info, accelerator):
    """
    为每个 group 生成共享的噪声，跨 GPU 同步。
    
    Args:
        x0: [batch_size, C, H, W] 用于获取形状
        group_info: 预计算的 group 信息，包含 inverse_indices, num_groups 等
        accelerator: Accelerator 对象
    
    Returns:
        noise_diffuse: [batch_size, C, H, W] 每个 group 内共享相同的噪声
    """
    batch_size = x0.shape[0]
    device = x0.device
    num_groups = group_info['num_groups']
    inverse_indices = group_info['inverse_indices']
    
    # 1. 只在 rank 0 生成噪声，然后广播
    if accelerator.is_main_process:
        group_noises = torch.randn(num_groups, *x0.shape[1:], device=device)
    else:
        group_noises = torch.empty(num_groups, *x0.shape[1:], device=device)
    
    # 广播噪声到所有 GPU
    dist.broadcast(group_noises, src=0)
    
    # 2. 根据 inverse_indices 把噪声分配给每个样本
    all_noises = group_noises[inverse_indices]  # [total_batch_size, C, H, W]
    
    # 3. 提取当前 GPU 的噪声
    noise_diffuse = all_noises[group_info['local_start']:group_info['local_end']]
    
    return noise_diffuse

def precompute_group_info(prompt_ids, accelerator):
    """用 prompt_ids 预计算 group 信息（更快）"""
    batch_size = prompt_ids.shape[0]
    
    local_group_ids = prompt_ids.view(batch_size, -1)
    all_group_ids = accelerator.gather(local_group_ids)
    
    _, inverse_indices = torch.unique(all_group_ids, dim=0, return_inverse=True)
    num_groups = inverse_indices.max().item() + 1
    
    rank = accelerator.process_index
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    
    local_group_indices = inverse_indices[start_idx:end_idx]
    
    return {
        'inverse_indices': inverse_indices,
        'local_group_indices': local_group_indices,
        'num_groups': num_groups,
        'local_start': start_idx,
        'local_end': end_idx,
        'batch_size': batch_size
    }

def verify_group_integrity(prompt_ids, accelerator, expected_group_size):
    """验证当前 mini-batch 中每个 group 是否恰好有 expected_group_size 个样本。"""
    batch_size = prompt_ids.shape[0]
    local_group_ids = prompt_ids.view(batch_size, -1)
    all_group_ids = accelerator.gather(local_group_ids)

    _, inverse_indices = torch.unique(all_group_ids, dim=0, return_inverse=True)
    num_groups = inverse_indices.max().item() + 1
    group_counts = torch.bincount(inverse_indices, minlength=num_groups)

    all_complete = (group_counts == expected_group_size).all().item()
    min_size = group_counts.min().item()
    max_size = group_counts.max().item()

    if accelerator.is_local_main_process:
        status = "PASS" if all_complete else "FAIL"
        print(f"[GroupVerify {status}] num_groups={num_groups}, "
              f"expected_size={expected_group_size}, "
              f"actual_min={min_size}, actual_max={max_size}")

    return {
        "group_verify_num_groups": num_groups,
        "group_verify_all_complete": int(all_complete),
        "group_verify_min_size": min_size,
        "group_verify_max_size": max_size,
    }

def compute_group_dgpo_loss_allreduce(
    model_v, ref_old_v, target_v, advantages,
    group_info, accelerator, beta_dpo, group_size = 24, dsm_loss = None,
):
    """AllReduce实现的梯度等价版本"""
    batch_size = model_v.shape[0]
    device = model_v.device
    
    if dsm_loss is None:
        dsm_loss = (target_v - model_v).square().reshape(batch_size, -1).mean(dim=1)
    with torch.no_grad():
        ref_dsm_loss = (target_v - ref_old_v).square().reshape(batch_size, -1).mean(dim=1)
    
    delta_diff = dsm_loss.detach() - ref_dsm_loss.detach()
    per_sample_term = advantages * beta_dpo * delta_diff / group_size
    
    local_group_indices = group_info['local_group_indices']
    num_groups = group_info['num_groups']
    
    local_group_sums = torch.zeros(num_groups, device=device, dtype=per_sample_term.dtype)
    local_group_sums.scatter_add_(0, local_group_indices, per_sample_term)
    
    global_group_sums = local_group_sums.clone().detach()
    dist.all_reduce(global_group_sums, op=dist.ReduceOp.SUM)
    
    # group_weights = 1 - torch.sigmoid(global_group_sums)
    group_weights = torch.sigmoid(global_group_sums)
    local_weights = group_weights[local_group_indices]
    
    loss = (local_weights.detach() * advantages * dsm_loss).mean()
    
    return loss


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(repeated_indices[start:end])
            yield per_card_samples[self.rank]
        # while True:
        #     # Generate a deterministic random sequence to ensure all replicas are synchronized
        #     g = torch.Generator()
        #     g.manual_seed(self.seed + self.epoch)
            
        #     # Randomly select m unique samples
        #     indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
        #     # Repeat each sample k times to generate n*b total samples
        #     repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
        #     # Shuffle to ensure uniform distribution
        #     shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
        #     shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
        #     # Split samples to each replica
        #     per_card_samples = []
        #     for i in range(self.num_replicas):
        #         start = i * self.batch_size
        #         end = start + self.batch_size
        #         per_card_samples.append(shuffled_samples[start:end])
            
        #     # Return current replica's sample indices
        #     yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def predict_v(transformer, noisy_samples, timesteps, embeds, pooled_embeds, config, cfg = True, cfg_scale = None):
    """
    修改后的函数：计算模型预测速度的对数概率。
    保持输入参数不变。
    """
    if cfg_scale is None:
        cfg_scale = config.sample.guidance_scale
    if config.train.cfg and cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([noisy_samples] * 2),
            timestep=torch.cat([timesteps] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond.detach()
        predicted_velocity = (
            noise_pred_uncond
            + cfg_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    elif config.train.cfg and (not cfg):
        embeds_uncond, embeds_cond = embeds.chunk(2)
        pooled_embeds_uncond, pooled_embeds_cond = pooled_embeds.chunk(2)
        predicted_velocity = transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds_cond,
            pooled_projections=pooled_embeds_cond,
            return_dict=False,
        )[0]
    else:
        predicted_velocity = transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    return predicted_velocity

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

    

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
            for key, value in {**{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()}}.items():
                metrics_history[key].append((global_step, value))

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    # num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    num_train_timesteps = config.num_train_timesteps

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        my_proj_name = f"sd3_dgpo-{config.sample.num_steps}steps-beta{config.train.beta}_{config.train.beta_dpo}-cfg{config.sample.guidance_scale}"
        my_proj_name +=  "_" + unique_id
        wandb.init(
            project="flow_dgpo",
            name = my_proj_name,
            mode="offline"  # 添加这行
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="scheduler")
    pipeline.scheduler.config['flow_shift'] = 3# the flow_shift can be changed from 1 to 6.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    ema_ref = EMAModuleWrapper(transformer_trainable_parameters, decay=0.3, update_step_interval=1, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_root = os.path.join(config.save_dir, "checkpoints", f"checkpoint-{global_step}")
            os.makedirs(save_root, exist_ok=True)
            save_ema_pth = os.path.join(save_root, "ema.ckpt")
            ema.save(save_ema_pth)
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        if global_step > config.switch_ema_ref:
            ema_ref.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents_list, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(
                latents_list, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "x0": latents_list[-1],
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )
        if global_step > config.switch_ema_ref:
            ema_ref.copy_temp_to(transformer_trainable_parameters)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                num_samples_new = min(16, len(images))  # 建议用16，正好4x4网格
                sample_indices_new = random.sample(range(len(images)), num_samples_new)
                selected_images = torch.stack([images[i] for i in sample_indices_new])
                grid = make_grid(
                        selected_images, 
                        nrow=4,  # 每行4张图片
                        padding=2,  # 图片间距
                        normalize=True,  # 自动归一化到[0,1]
                        value_range=(0, 1)  # 如果你的图片已经在[0,1]范围内
                    )
                save_dir = os.path.join(config.logdir, my_proj_name)
                os.makedirs(save_dir, exist_ok=True)
                save_image(grid, os.path.join(save_dir, f"grid_epoch_{epoch}.jpg") )



                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.num_steps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            reward_metrics = {f"reward_{key}": value.mean() for key, value in gathered_rewards.items() 
                     if '_strict_accuracy' not in key and '_accuracy' not in key}
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )
            for key, value in reward_metrics.items():
                metrics_history[key].append((global_step, float(value)))


        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        # del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        # samples = {k: v[mask] for k, v in samples.items()}
        samples = {k: v for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        use_old = config.clip_dsm or config.clip_kl
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)

            M = num_train_timesteps
            train_timesteps = [step_index for step_index in range(config.trunc_steps + 1)]
            sampled_timesteps = generate_shared_sampled_timesteps(accelerator, train_timesteps, M)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]
                group_info = precompute_group_info(sample["prompt_ids"], accelerator)
                # group_verify_info = verify_group_integrity(
                #     sample["prompt_ids"], accelerator, config.sample.num_image_per_prompt
                # )
                # for gv_k, gv_v in group_verify_info.items():
                #     info[gv_k].append(torch.tensor(gv_v, device=accelerator.device, dtype=torch.float32))
                
                for j in tqdm(
                    sampled_timesteps,
                    total=M,  # 明确告诉 tqdm 总的迭代次数是 M
                    desc="Sampling Timesteps",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        x0 = sample["x0"]
                        t = sample["timesteps"][:, j]
                        sigmas_t = (t / 1000).reshape(x0.shape[0],1,1,1) 

                        if config.use_shared_noise:
                            noise_diffuse = generate_shared_noise_for_groups(x0, group_info, accelerator)
                        else:
                            noise_diffuse = torch.randn_like(x0)
                        xt = (1 - sigmas_t) * x0 + sigmas_t * noise_diffuse
                        target_v = noise_diffuse - x0

                        with autocast():
                            with torch.no_grad():
                                with transformer.module.disable_adapter():
                                    if config.kl_cfg > 1:
                                        ref_old_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg = True, cfg_scale=config.kl_cfg)
                                    else:
                                        ref_old_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg = False)
                                if use_old:
                                    ema_ref.copy_ema_to(transformer_trainable_parameters, store_temp=True)
                                    old_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg = False)
                                    ema_ref.copy_temp_to(transformer_trainable_parameters)
                            model_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg = False)

                        ref_dgpo_v = ref_old_v
                        if config.use_ema_ref:
                            ref_dgpo_v = old_v

                        # grpo-style advantages
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        ) # For numerical stability, the requirement that \sum_G A = 0 is slightly violated. But clipping rarely happens in practice.
                        dsm_loss = 1 * (target_v - model_v).square().reshape(x0.shape[0],-1).mean(dim=1)
                        if use_old:
                            old_dsm_loss = 1 * (target_v - old_v).square().reshape(x0.shape[0],-1).mean(dim=1)
                            # ppo-style clipping
                            ratio = torch.exp(-dsm_loss + old_dsm_loss)
                            clip_range = config.clip_range
                            should_clip = torch.where(
                                advantages > 0,
                                ratio > 1.0 + clip_range, 
                                ratio < 1.0 - clip_range, 
                            )
                            if config.clip_dsm:
                                dsm_loss = torch.where(should_clip, dsm_loss.detach(), dsm_loss)

                        dgpo_loss = compute_group_dgpo_loss_allreduce(
                            model_v, ref_dgpo_v, target_v, advantages,
                            group_info, accelerator, config.train.beta_dpo, group_size=config.sample.num_image_per_prompt, dsm_loss = dsm_loss,
                        )
                        if config.train.beta > 0:
                            kl_loss = 1 * (model_v - ref_old_v).square().reshape(x0.shape[0],-1).mean(dim=1)
                            if config.clip_kl:
                                kl_loss = torch.where(should_clip, kl_loss.detach(), kl_loss)
                            kl_loss = torch.mean(kl_loss)
                            info["kl_loss"].append(kl_loss)
                            loss = dgpo_loss + config.train.beta * kl_loss
                        else:
                            loss = dgpo_loss

                        if use_old:
                            clip_ratio_total = should_clip.float().mean()
                            info["clip_ratio_total"].append(clip_ratio_total)

                        info["policy_loss"].append(dgpo_loss)
                        info["dsm_loss"].append(dsm_loss.mean())
                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )

                        optimizer.step()
                        optimizer.zero_grad()
                        if accelerator.sync_gradients:
                            ema_ref_decay = min(0.3, 0.001 * global_step)
                            ema_ref.step(transformer_trainable_parameters, global_step, decay = ema_ref_decay)


                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                            
                            # 更新 metrics_history（修复缩进）
                            for key, value in info.items():
                                if key not in ["epoch", "inner_epoch"]:
                                    metrics_history[key].append((global_step, float(value)))
                            
                            # 保存图片
                            save_dir = os.path.join(config.logdir, my_proj_name)
                            os.makedirs(save_dir, exist_ok=True)
                            print(save_dir)
                            
                            for metric_name, values in metrics_history.items():
                                if len(values) > 1:
                                    steps, metric_values = zip(*values)
                                    plt.figure(figsize=(10, 6))
                                    plt.plot(steps, metric_values)
                                    plt.title(f'{metric_name} over time')
                                    plt.xlabel('Global Step')
                                    plt.ylabel(metric_name)
                                    plt.grid(True)
                                    plt.savefig(os.path.join(save_dir, f'{metric_name}.jpg'), 
                                            dpi=150, bbox_inches='tight')
                                    plt.close()
                        global_step += 1
                        info = defaultdict(list)


                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)