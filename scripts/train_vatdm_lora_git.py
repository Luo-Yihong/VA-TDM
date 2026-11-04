from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.rewards
from flow_grpo.diffusers_patch.sd3_pipeline_vatdm import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

from torchvision.utils import make_grid, save_image
from diffusers import SD3Transformer2DModel
from flow_grpo.diffusers_patch.sd3_model_patch_v2 import (
    sd3_transformer_forward_joint_v2,
    create_zero_init_time_text_embed,
    create_joint_output_heads,
)

SD3Transformer2DModel.forward = sd3_transformer_forward_joint_v2

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def convert(x, s=7):
    return s * x / 1000 / (1 + (s-1) * x / 1000) * 1000

def inv_convert(y, s=7):
    return 1000 * y / (1000 * s - (s - 1) * y)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


@contextlib.contextmanager
def score_adapter(transformer, adapter=None):
    """Freeze score parameters, retaining the derivative with respect to inputs.

    Adapter switching changes requires_grad in PEFT; restore both it and the
    active adapter before the generator backward. Checkpoint recomputation must
    preserve its own forward-time adapter and trainability state.
    """
    active_adapter = transformer.active_adapter
    parameters = list(transformer.parameters())
    requires_grad = [p.requires_grad for p in parameters]
    try:
        if adapter is not None:
            transformer.set_adapter(adapter)
        transformer.requires_grad_(False)
        with transformer.disable_adapter() if adapter is None else contextlib.nullcontext():
            yield
    finally:
        transformer.set_adapter(active_adapter)
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


def step_optimizer(optimizer, parameters, scaler, accelerator, max_grad_norm):
    """Synchronize only this optimizer's accumulated gradients, then step.

    Two adapters share one backbone and change trainability between forwards.
    Explicit reduction avoids DDP's fixed parameter/reducer assumptions and
    keeps fake and generator gradient scaling independent.
    """
    if not accelerator.sync_gradients:
        return
    if accelerator.num_processes > 1:
        used = torch.tensor([p.grad is not None for p in parameters],
                            device=accelerator.device, dtype=torch.int32)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        active = [p for p, present in zip(parameters, used.tolist()) if present]
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in active])
        dist.all_reduce(flat)
        flat.div_(accelerator.num_processes)
        offset = 0
        for p in active:
            p.grad = flat[offset:offset + p.numel()].view_as(p)
            offset += p.numel()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def enable_adapter_checkpointing(transformer):
    """Recompute each block with the adapter/freeze state of its forward."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from torch.utils.checkpoint import checkpoint

    def checkpoint_block(block, *args):
        layers = [m for m in block.modules() if isinstance(m, BaseTunerLayer)]
        parameters = list(block.parameters())

        def capture():
            return ([(list(m.active_adapters), m.disable_adapters) for m in layers],
                    [p.requires_grad for p in parameters])

        def restore(state):
            adapter_states, grad_states = state
            for module, (adapters, disabled) in zip(layers, adapter_states):
                module.set_adapter(adapters)
                module.enable_adapters(not disabled)
            for parameter, enabled in zip(parameters, grad_states):
                parameter.requires_grad_(enabled)

        forward_state = capture()

        @contextlib.contextmanager
        def recompute():
            previous = capture()
            restore(forward_state)
            try:
                yield
            finally:
                restore(previous)

        return checkpoint(block, *args, use_reentrant=False,
                          context_fn=lambda: (contextlib.nullcontext(), recompute()))

    transformer.get_base_model().enable_gradient_checkpointing(
        gradient_checkpointing_func=checkpoint_block,
    )


def collate_prompts(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    return prompts, metadatas


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}


PROMPT_DATASETS = {"pickscore": TextPromptDataset}


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
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            # Return current replica's sample indices
            start = self.rank * self.batch_size
            yield shuffled_samples[start:start + self.batch_size]

    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def predict_v_joint(transformer, noisy_samples, timesteps, embeds, pooled_embeds, config, target_timestep):
    """Predict fake [score, moment] velocities with one shared trunk/time and independent heads."""
    if config.train.cfg:
        embeds = embeds.chunk(2)[1]
        pooled_embeds = pooled_embeds.chunk(2)[1]
    predicted_velocity, predicted_velocity_moment = transformer(
        hidden_states=noisy_samples,
        timestep=timesteps,
        use_joint_heads=True,
        target_timestep=target_timestep,
        return_both=True,
        encoder_hidden_states=embeds,
        pooled_projections=pooled_embeds,
        return_dict=False,
    )
    return predicted_velocity[0], predicted_velocity_moment[0]


def predict_v(transformer, noisy_samples, timesteps, embeds, pooled_embeds, config, cfg = True, cfg_scale = None):
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
        return noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
    if config.train.cfg:
        embeds = embeds.chunk(2)[1]
        pooled_embeds = pooled_embeds.chunk(2)[1]
    return transformer(
        hidden_states=noisy_samples,
        timestep=timesteps,
        encoder_hidden_states=embeds,
        pooled_projections=pooled_embeds,
        return_dict=False,
    )[0]


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        generators.append(torch.Generator().manual_seed(seed))
    return generators


def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, autocast, ema, transformer_trainable_parameters):
    pipeline.transformer.set_adapter("tdm")
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

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
                )
        rewards, _ = reward_fn(images, prompts, prompt_metadata)

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
            sample_indices = range(min(15, len(last_batch_images_gather)))
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items()),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value) for key, value in all_rewards.items()},
                },
                step=global_step,
            )

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
        unwrapped = unwrap_model(transformer, accelerator)
        unwrapped.save_pretrained(save_root_lora)
        # Persist the zero-init second time embedding (2-timestep fake score module).
        base_model = unwrapped
        for attr in ("base_model", "model"):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
        tte2_module = getattr(base_model, "time_text_embed_2", None)
        if tte2_module is not None:
            torch.save(tte2_module.state_dict(), os.path.join(save_root, "time_text_embed_2.pt"))
        joint_heads = getattr(base_model, "joint_heads", None)
        if joint_heads is not None:
            torch.save(joint_heads.state_dict(), os.path.join(save_root, "joint_heads.pt"))
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
    num_train_timesteps = 2

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        base_name = "vatdm-huber-shift" if config.use_huber else "vatdm-mse-shift"
        resume_tag = "_ct" if config.train.lora_path else ""
        my_proj_name = (
            f"{base_name}{resume_tag}_ablation-{config.sample.num_steps}steps"
            f"-w_moment_{config.lambda_moment}-r2_{config.lambda_reg_r2}-ttur{config.ttur}"
            f"_b{config.lambda_b}_{unique_id}"
        )
        save_dir = os.path.join(config.logdir, my_proj_name)
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "vatdm-lora-git"),
            config=config.to_dict(),
            name = my_proj_name,
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    if config.use_dpm:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="scheduler")
        pipeline.scheduler.config['flow_shift'] = config.flow_shift# the flow_shift can be changed from 1 to 6.
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.vae.enable_slicing()
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

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

    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
    # Enable state-preserving checkpointing after the adapters are installed.
    pipeline.transformer.disable_gradient_checkpointing()

    # Zero-init second time embedding used by the 2-timestep fake score model.
    # Attach BEFORE LoRA wrapping so the module is shared across all adapters.
    tte2 = create_zero_init_time_text_embed(pipeline.transformer)
    tte2.to(accelerator.device, dtype=torch.float32)
    pipeline.transformer.time_text_embed_2 = tte2
    pipeline.transformer.joint_heads = create_joint_output_heads(pipeline.transformer).float()

    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ],
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="tdm")
    pipeline.transformer.add_adapter("fake", transformer_lora_config)
    # FP16/BF16 frozen backbone, FP32 adapter parameters and optimizer updates.
    for name, parameter in pipeline.transformer.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    enable_adapter_checkpointing(pipeline.transformer)

    pipeline.transformer.set_adapter("tdm")
    transformer = pipeline.transformer
    transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "tdm" in name:
            assert param.requires_grad == True
            transformer_trainable_parameters.append(param)

    pipeline.transformer.set_adapter("fake")
    fake_transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "fake" in name:
            assert param.requires_grad == True
            fake_transformer_trainable_parameters.append(param)

    tte2_params = [p for n, p in transformer.named_parameters() if "time_text_embed_2" in n]
    joint_head_params = [p for n, p in transformer.named_parameters() if "joint_heads." in n]
    fake_extra_parameters = tte2_params + joint_head_params
    for parameter in fake_extra_parameters:
        parameter.requires_grad_(True)
    fake_parameters = fake_transformer_trainable_parameters + fake_extra_parameters

    # Rank-specific sampling seeds must not give different LoRA initial weights.
    if accelerator.num_processes > 1:
        for parameter in transformer.parameters():
            dist.broadcast(parameter.data, src=0)
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.99, update_step_interval=1, device=accelerator.device)
    if config.train.lora_path:
        print(f"Loading from {config.train.lora_path}")
        ema.load(f"{config.train.lora_path}")
        ema.copy_ema_to(transformer_trainable_parameters)

    accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
    scaler_fake = torch.amp.GradScaler("cuda", enabled=accelerator.mixed_precision == "fp16")
    scaler_tdm = torch.amp.GradScaler("cuda", enabled=accelerator.mixed_precision == "fp16")

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
        lr=config.train.learning_rate_generator,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    optimizer_fake = optimizer_cls(
        [
            {
                "params": fake_transformer_trainable_parameters,
                "lr": config.train.learning_rate_fake,
            },
            {
                "params": fake_extra_parameters,
                "lr": 2e-5,
            },
        ],
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = flow_grpo.rewards.multi_score(accelerator.device, config.reward_fn)

    if config.prompt_fn not in PROMPT_DATASETS:
        raise NotImplementedError(f"Unsupported prompt_fn: {config.prompt_fn}")
    dataset_cls = PROMPT_DATASETS[config.prompt_fn]
    train_dataset = dataset_cls(config.dataset, 'train')
    test_dataset = dataset_cls(config.dataset, 'test')

    # Infinite sampler; shuffling is controlled by the sampler, not the DataLoader.
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
        num_workers=1,
        collate_fn=collate_prompts,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collate_prompts,
        shuffle=False,
        num_workers=8,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # Keep trainable adapter weights in FP32 and autocast model forwards.
    autocast = accelerator.autocast

    # The training sampler already shards by rank. Only shard the eval loader;
    # optimizers use separate scalers and explicit adapter-gradient reduction.
    test_dataloader = accelerator.prepare(test_dataloader)

    # Rewards are computed asynchronously so the next sampling batch can start.
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

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    max_epochs = getattr(config, "num_epochs", float("inf"))
    while epoch < max_epochs:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator,
                 global_step, reward_fn, autocast, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            pipeline.transformer.set_adapter("tdm")
            for root in (config.save_dir, save_dir):
                save_root = os.path.join(root, "checkpoints", f"checkpoint-{global_step}")
                os.makedirs(save_root, exist_ok=True)
                ema.save(os.path.join(save_root, "ema.ckpt"))
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        samples = []
        reward_futures = []
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

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            pipeline.transformer.set_adapter("tdm")
            with autocast():
                with torch.no_grad():
                    images, latents_list, _ = pipeline_with_logprob(
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
                        generator=generator
                )

            latents = torch.stack(latents_list, dim=1)  # (batch_size, num_steps + 1, 16, 96, 96)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.train_batch_size, 1)
            zeros_t = torch.zeros(
                config.sample.train_batch_size, 1, dtype=timesteps.dtype, device=timesteps.device
            )
            timesteps = torch.cat([timesteps, zeros_t], dim=1)  # (batch_size, num_steps + 1)

            reward_futures.append(
                executor.submit(reward_fn, images, prompts, prompt_metadata)
            )
            # yield to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "x0": latents_list[-1],
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                }
            )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples], dim=0) for k in samples[0]}

        epoch_rewards = defaultdict(list)
        for future in tqdm(
            reward_futures,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            last_batch_rewards, _ = future.result()
            for key, value in last_batch_rewards.items():
                epoch_rewards[key].append(torch.as_tensor(value, device=accelerator.device).float())
        reward_metrics = {
            f"reward_{key}": accelerator.gather(torch.cat(value)).mean().item()
            for key, value in epoch_rewards.items()
        }
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **reward_metrics,
                },
                step=global_step,
            )

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
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sample_indices_new = random.sample(range(len(images)), min(16, len(images)))
                selected_images = torch.stack([images[i] for i in sample_indices_new])
                grid = make_grid(selected_images, nrow=4, padding=2, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(save_dir, f"grid_epoch_{epoch}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [last_batch_rewards['avg'][i] for i in sample_indices]

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

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps + 1 # Include the zero timestep

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

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

                sampled_timesteps = random.sample(range(config.sample.num_steps), num_train_timesteps)
                for j in tqdm(
                    sampled_timesteps,
                    desc="Sampling Timesteps",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        x0 = sample["x0"]
                        t = sample["timesteps"][:, j].float()
                        t_next = sample["timesteps"][:, j+1].float() # [0, 1000]
                        xt = sample["latents"][:, j]

                        lower_bound_inv = inv_convert(t_next + 20, s=3)
                        upper_bound_inv = inv_convert(980, s=3)
                        tau_inv = torch.rand_like(t_next) * (upper_bound_inv - lower_bound_inv) + lower_bound_inv
                        tau = convert(tau_inv, s=3)

                        sigmas_t = (t / 1000).reshape(x0.shape[0],1,1,1)
                        sigmas_tau = (tau / 1000).reshape(x0.shape[0],1,1,1)
                        sigmas_tnext = (t_next / 1000).reshape(x0.shape[0],1,1,1)

                        # Train fake score
                        with autocast():
                            pipeline.transformer.set_adapter("tdm")
                            with torch.no_grad():
                                model_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg = False)
                                model_x0 = xt - sigmas_t * model_v
                                model_xnext = xt - (sigmas_t - sigmas_tnext) * model_v
                            alpha = (1 - sigmas_tau) / (1 - sigmas_tnext)
                            beta = (sigmas_tau ** 2 - (sigmas_tnext * alpha ) ** 2) ** 0.5
                            noise_diffuse = torch.randn_like(model_xnext)
                            xtau = model_xnext * alpha + beta * noise_diffuse
                            target_v = (xtau - model_x0) / sigmas_tau

                            pipeline.transformer.set_adapter("fake")
                            for parameter in fake_extra_parameters:
                                parameter.requires_grad_(True)
                            fake_v, fake_v_moment = predict_v_joint(
                                transformer, xtau, tau, embeds, pooled_embeds, config,
                                target_timestep=t_next,
                            )
                            if config.lambda_reg_r2 > 0:
                                with torch.no_grad():
                                    with score_adapter(transformer):
                                        real_v = predict_v(transformer, xtau, tau, embeds, pooled_embeds, config, cfg = False)
                                loss_reg_score = (real_v.detach() - fake_v).square().mean()
                                loss_reg_moment = (real_v.detach() - fake_v_moment).square().mean()
                                loss_reg = config.lambda_moment * loss_reg_moment + (1 - config.lambda_moment) * loss_reg_score
                            target_v_score = - model_xnext / (1 - sigmas_tnext) + noise_diffuse / beta * (sigmas_tau + sigmas_tnext ** 2 * alpha / (1 - sigmas_tnext))
                            loss_fake_moment = ((fake_v_moment - target_v.detach()).square())
                            loss_fake_score = ((fake_v - target_v_score.detach()).square())
                            loss_fake = config.lambda_moment * loss_fake_moment + (1 - config.lambda_moment) * loss_fake_score
                            loss_fake = loss_fake.mean()
                        fake_loss_step = loss_fake
                        if config.lambda_reg_r2 > 0:
                            fake_loss_step = fake_loss_step + config.lambda_reg_r2 * loss_reg
                        scaler_fake.scale(fake_loss_step / accum_steps).backward()
                        step_optimizer(optimizer_fake, fake_parameters, scaler_fake,
                                       accelerator, config.train.max_grad_norm)

                        # Train generator
                        lower_bound_inv = inv_convert(t_next, s=3).clamp(min=20)
                        tau_tdm_inv = torch.rand_like(t_next) * (upper_bound_inv - lower_bound_inv) + lower_bound_inv
                        tau_tdm = convert(tau_tdm_inv, s=3)

                        sigmas_tau_tdm = (tau_tdm / 1000).reshape(x0.shape[0],1,1,1)
                        alpha_tdm = (1 - sigmas_tau_tdm) / (1 - sigmas_tnext)
                        beta_tdm = (sigmas_tau_tdm ** 2 - (sigmas_tnext * alpha_tdm ) ** 2) ** 0.5

                        if global_step % config.ttur == 0:
                            pipeline.transformer.set_adapter("tdm")
                            for parameter in fake_extra_parameters:
                                parameter.requires_grad_(False)
                            with autocast():
                                model_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg=False)
                                model_x0 = xt - sigmas_t * model_v
                            model_xnext = xt - (sigmas_t - sigmas_tnext) * model_v
                            noise_tdm = torch.randn_like(model_xnext)
                            xtau_tdm = model_xnext * alpha_tdm + beta_tdm * noise_tdm
                            cfg_rescale = 1 / (1 + config.lambda_reg_r2) # Correct cfg scale
                            with autocast():
                                with score_adapter(transformer, "fake"):
                                    fake_v, fake_v_moment = predict_v_joint(
                                        transformer, xtau_tdm, tau_tdm, embeds, pooled_embeds, config,
                                        target_timestep=t_next,
                                    )
                                with score_adapter(transformer):
                                    if config.lambda_reg_r2 > 0:
                                        real_cond_v = predict_v(transformer, xtau_tdm, tau_tdm, embeds,
                                                               pooled_embeds, config, cfg=False)
                                        real_cond_x0 = xtau_tdm.detach() - real_cond_v * sigmas_tau_tdm
                                    with torch.no_grad():
                                        real_v = predict_v(transformer, xtau_tdm, tau_tdm, embeds, pooled_embeds,
                                                           config, cfg=True, cfg_scale=1 + (config.cfg_scale - 1) * cfg_rescale)
                                fake_x0 = xtau_tdm.detach() - fake_v * sigmas_tau_tdm
                                fake_x0_moment = xtau_tdm.detach() - fake_v_moment * sigmas_tau_tdm
                                real_x0 = xtau_tdm.detach() - real_v * sigmas_tau_tdm

                            weighting_factor = torch.abs(model_x0.double() - real_x0.double()).mean(dim=[1, 2, 3], keepdim=True).detach()
                            weighting_factor = weighting_factor.clip(max=50, min=0.02)
                            target_direction = (real_x0 - fake_x0).detach()
                            target_direction_moment = (real_x0 - fake_x0_moment).detach()
                            fake_x0_correct = fake_x0
                            fake_x0_moment_correct = fake_x0_moment
                            if config.lambda_reg_r2 > 0:
                                # Correct fake prediction.
                                fake_x0_correct = (
                                    (1 + config.lambda_reg_r2) * fake_x0
                                    - config.lambda_reg_r2 * real_cond_x0
                                )
                                fake_x0_moment_correct = (
                                    (1 + config.lambda_reg_r2) * fake_x0_moment
                                    - config.lambda_reg_r2 * real_cond_x0
                                )
                            # The loss is reweighted to be computed in x0 sapce rather than v space.
                            loss_vad_score = target_direction * (config.lambda_b * fake_x0_correct - model_x0)
                            loss_vad_moment = target_direction_moment * (config.lambda_b * fake_x0_moment_correct - model_x0)
                            if config.use_huber:
                                loss_vad_score = loss_vad_score / torch.sqrt(target_direction.square() + 1e-3)
                                loss_vad_moment = loss_vad_moment / torch.sqrt(target_direction_moment.square() + 1e-3)
                            loss = config.lambda_moment * loss_vad_moment + (1 - config.lambda_moment) * loss_vad_score
                            loss = loss / weighting_factor
                            loss = loss.mean()

                            scaler_tdm.scale(loss / accum_steps).backward()
                            step_optimizer(optimizer, transformer_trainable_parameters, scaler_tdm,
                                           accelerator, config.train.max_grad_norm)
                            if accelerator.sync_gradients and config.train.ema:
                                ema.step(transformer_trainable_parameters, global_step, decay=0.99)
                            info["loss"].append(loss.detach())

                        info["fake_loss"].append(loss_fake.detach())
                        if config.lambda_reg_r2 > 0:
                            info["fake_r2_reg_loss"].append(loss_reg.detach())

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            with torch.no_grad():
                                images_4step = pipeline.vae.decode(x0[:4].to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)*0.5+0.5
                            save_image(images_4step, os.path.join(save_dir, 'x0_samples.jpg'), normalize = False, nrow = 2)
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

        epoch+=1

    if accelerator.is_main_process:
        pipeline.transformer.set_adapter("tdm")
        save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
    executor.shutdown(wait=False)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()
    # Exit immediately so lingering DataLoader workers cannot hang the job.
    os._exit(0)

if __name__ == "__main__":
    app.run(main)
