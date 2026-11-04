"""OT-ODE warm-up, compatible with the VATDM student LoRA."""
from config.vatdm import pickscore_tdm2_8gpu_G64


def get_config():
    config = pickscore_tdm2_8gpu_G64()
    config.run_name = "otode-lora"
    config.mixed_precision = "fp16"
    config.train.learning_rate_generator = 2e-4
    config.train.adam_beta2 = 0.95
    config.save_dir = "logs/otode"
    # One optimizer update per epoch: 2 rollout batches x 2 intervals = 4 accumulation steps.
    # 8 GPUs x 16 x 2 batches = 256 images (16 prompts x 16 samples) per update.
    config.num_epochs = 601
    config.sample.train_batch_size = 16
    config.train.batch_size = 16
    config.sample.num_image_per_prompt = 16
    config.eval_freq = 60
    config.save_freq = 60
    config.otode = dict(
        teacher_solver="euler",  # euler | heun | dpm
        teacher_substeps=8,
        teacher_guidance_scale=4.5,
        ema_decay=0.9,
    )
    return config
