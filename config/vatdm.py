"""Eight-GPU PickScore VATDM preset."""
import os
from pathlib import Path
import ml_collections


def pickscore_tdm2_8gpu_G64():
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    model_root = Path(os.environ.get("VATDM_MODEL_ROOT", workspace / "models" / "vatdm"))
    config = ml_collections.ConfigDict({'allow_tf32': True,
     'cfg_scale': 4.5,
     'dataset': str(root / "dataset" / "pickscore"),
     'eval_freq': 60,
     'flow_shift': 3,
     'lambda_b': 0.7,
     'lambda_moment': 0.7,
     'lambda_reg_r2': 0.5,
     'logdir': 'logs',
     'mixed_precision': 'bf16',
     'num_checkpoint_limit': 5,
     'num_epochs': 1021,
     'pretrained': {'model': str(model_root / "stable-diffusion-3.5-medium")},
     'prompt_fn': 'pickscore',
     'resolution': 512,
     'reward_fn': {'pickscore': 1.0},
     'run_name': '',
     'sample': {'eval_num_steps': 4,
                'guidance_scale': 1.0,
                'num_batches_per_epoch': 2,
                'num_image_per_prompt': 12,
                'num_steps': 4,
                'same_latent': False,
                'test_batch_size': 16,
                'train_batch_size': 12},
     'save_dir': 'logs/pickscore/tdm2_dpm',
     'save_freq': 60,
     'seed': 42,
     'train': {'adam_beta1': 0.0,
               'adam_beta2': 0.95,
               'adam_epsilon': 1e-08,
               'adam_weight_decay': 0.0001,
               'batch_size': 12,
               'cfg': True,
               'ema': True,
               'gradient_accumulation_steps': 2,
               'learning_rate_fake': 2e-4,
               'learning_rate_generator': 5e-5,
               'lora_path': '',
               'max_grad_norm': 1.0,
               'num_inner_epochs': 1,
               'use_8bit_adam': False},
     'ttur': 1,
     'use_dpm': True,
     'use_huber': True})
    return config


def get_config(name):
    if name != "pickscore_tdm2_8gpu_G64":
        raise ValueError(f"Unsupported VATDM config: {name}")
    return pickscore_tdm2_8gpu_G64()
