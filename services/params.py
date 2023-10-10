from typing import Dict


def create_dreambooth_params(event: Dict, model_path: str, cog_output_dir: str, train_data_dir: str, ckpt_base: str) -> Dict:
    seed = event['input'].get("seed", 1337)
    resolution = event['input'].get("resolution", 512)
    train_batch_size = event['input'].get("train_batch_size", 1)
    max_train_steps = event['input'].get('max_train_steps', 880)
    learning_rate = event['input'].get('learning_rate', 1e-6)
    lr_scheduler = event['input'].get("lr_scheduler", 'constant')
    lr_warmup_steps = event['input'].get('lr_warmup_steps', 0)
    use_8bit_adam = event['input'].get('use_8bit_adam', False)
    adaptive_noise_scale = event['input'].get('adaptive_noise_scale', 0.015)
    bucket_no_upscale = event['input'].get('bucket_no_upscale', False)
    bucket_reso_steps = event['input'].get('bucket_reso_steps', 64)
    clip_skip = event['input'].get('clip_skip', 1)
    optimizer_type = event['input'].get('optimizer_type', "AdamW8bit")
    color_aug = event['input'].get('color_aug', False)
    prior_loss_weight = event['input'].get('prior_loss_weight', 0.75)
    sample_sampler = event['input'].get('sample_sampler', 'dpmsolver++')
    output_name = event['input'].get('model_id', 'bra5_dreambooth')
    # some settings are fixed for the replicate model
    vae = f"{model_path}/vae/vae-ft-mse-840000-ema-pruned.safetensors"
    args = {
        'v2': False,
        'v_parameterization': False,
        'pretrained_model_name_or_path': ckpt_base,
        'tokenizer_cache_dir': None,
        'train_data_dir': train_data_dir,
        'shuffle_caption': True,
        'caption_extension': '.txt',
        'caption_extention': None,
        'keep_tokens': 0,
        'color_aug': color_aug,
        'flip_aug': False,
        'face_crop_aug_range': None,
        'random_crop': False,
        'debug_dataset': False,
        'resolution': f'{resolution},{resolution}',
        'cache_latents': True,
        'vae_batch_size': 1,
        'cache_latents_to_disk': False,
        'enable_bucket': True,
        'min_bucket_reso': 256,
        'max_bucket_reso': 1024,
        'bucket_reso_steps': bucket_reso_steps,
        'bucket_no_upscale': bucket_no_upscale,
        'token_warmup_min': 1,
        'token_warmup_step': 0,
        'dataset_class': None,
        'caption_dropout_rate': 0.03,
        'caption_dropout_every_n_epochs': 0,
        'caption_tag_dropout_rate': 0.0,
        'reg_data_dir': None,
        'output_dir': cog_output_dir,
        'output_name': output_name,
        'huggingface_repo_id': None,
        'huggingface_repo_type': None,
        'huggingface_path_in_repo': None,
        'huggingface_token': None,
        'huggingface_repo_visibility': None,
        'save_state_to_huggingface': False,
        'resume_from_huggingface': False,
        'async_upload': False,
        'save_precision': 'fp16',
        'save_every_n_epochs': 1,
        'save_every_n_steps': None,
        'save_n_epoch_ratio': None,
        'save_last_n_epochs': None,
        'save_last_n_epochs_state': None,
        'save_last_n_steps': None,
        'save_last_n_steps_state': None,
        'save_state': False,
        'resume': None,
        'train_batch_size': train_batch_size,
        'max_token_length': 225,
        'mem_eff_attn': False,
        'xformers': True,
        'vae': vae,
        'max_train_steps': max_train_steps,
        'max_train_epochs': None,
        'max_data_loader_n_workers': 0,
        'persistent_data_loader_workers': False,
        'seed': seed,
        'gradient_checkpointing': False,
        'gradient_accumulation_steps': 1,
        'mixed_precision': 'fp16',
        'full_fp16': False,
        'clip_skip': clip_skip,
        'logging_dir': None,
        'log_with': None,
        'log_prefix': None,
        'log_tracker_name': None,
        'wandb_api_key': None,
        'noise_offset': 0.15,
        'multires_noise_iterations': None,
        'multires_noise_discount': 0.3,
        'adaptive_noise_scale': adaptive_noise_scale,
        'lowram': False,
        'sample_every_n_steps': None,
        'sample_every_n_epochs': None,
        'sample_prompts': None,
        'sample_sampler': sample_sampler,
        'config_file': None,
        'output_config': False,
        'prior_loss_weight': prior_loss_weight,
        'save_model_as': "diffusers",
        'use_safetensors': False,
        'optimizer_type': optimizer_type,
        'use_8bit_adam': use_8bit_adam,
        'use_lion_optimizer': False,
        'learning_rate': learning_rate,
        'max_grad_norm': 1.0,
        'optimizer_args': None,
        'lr_scheduler_type': '',
        'lr_scheduler_args': None,
        'lr_scheduler': lr_scheduler,
        'lr_warmup_steps': lr_warmup_steps,
        'lr_scheduler_num_cycles': 1,
        'lr_scheduler_power': 1,
        'dataset_config': None,
        'min_snr_gamma': 1.0,
        'scale_v_pred_loss_like_noise_pred': False,
        'weighted_captions': True,
        'no_token_padding': False,
        'stop_text_encoder_training': None
    }
    return args


def create_lora_params(event: Dict, cog_output_dir: str, train_data_dir: str, ckpt_base: str, image_size_count: int) -> Dict:
    seed = event['input'].get("seed", 1337)
    resolution = event['input'].get("resolution", 512)
    train_batch_size = event['input'].get("train_batch_size", 2)
    max_train_steps = event['input'].get("max_train_steps", 880)
    network_alpha = event['input'].get('network_alpha', 32)
    network_module = event['input'].get('network_module', 'lycoris.kohya')
    network_args = event['input'].get('network_args', ['conv_dim=32', 'conv_alpha=32', 'algo=lora'])
    learning_rate = event['input'].get('learning_rate', 2e-05)
    save_every_epoches = event['input'].get('learning_rate', 2e-05)

    lr_scheduler = event['input'].get("lr_scheduler", 'constant_with_warmup')
    lr_warmup_steps = int(max_train_steps / 10)
    use_8bit_adam = event['input'].get('use_8bit_adam', False)
    adaptive_noise_scale = event['input'].get('adaptive_noise_scale', 0.01)
    bucket_no_upscale = event['input'].get('bucket_no_upscale', False)
    bucket_reso_steps = event['input'].get('bucket_reso_steps', 64)
    clip_skip = event['input'].get('clip_skip', 1)
    optimizer_type = event['input'].get('optimizer_type', "AdamW8bit")
    color_aug = event['input'].get('color_aug', False)
    prior_loss_weight = event['input'].get('prior_loss_weight', 0.75)
    sample_sampler = event['input'].get('sample_sampler', 'dpmsolver++')
    output_name = event['input'].get('model_id', 'testmodel')
    min_bucket_reso = event['input'].get('min_bucket_reso', 64)
    max_bucket_reso = event['input'].get('max_bucket_reso', 2048)
    args = {
        'v2': False,
        'v_parameterization': False,
        'network_alpha': network_alpha,
        'pretrained_model_name_or_path': ckpt_base,
        'tokenizer_cache_dir': None,
        'train_data_dir': train_data_dir,
        'shuffle_caption': True,
        'caption_extension': '.txt',
        'caption_extention': None,
        'keep_tokens': 0,
        'color_aug': color_aug,
        'flip_aug': False,
        'face_crop_aug_range': None,
        'random_crop': False,
        'debug_dataset': False,
        'resolution': f'{resolution},{resolution}',
        'cache_latents': True,
        'vae_batch_size': 1,
        'enable_bucket': True,
        'min_bucket_reso': min_bucket_reso,
        'max_bucket_reso': max_bucket_reso,
        'bucket_reso_steps': bucket_reso_steps,
        'bucket_no_upscale': bucket_no_upscale,
        'caption_dropout_rate': 0.03,
        'caption_tag_dropout_rate': 0.0,
        'reg_data_dir': None,
        'output_dir': cog_output_dir,
        'output_name': output_name,
        'huggingface_repo_id': None,
        'huggingface_repo_type': None,
        'huggingface_path_in_repo': None,
        'huggingface_token': None,
        'cache_latents_to_disk': True,
        'huggingface_repo_visibility': None,
        'save_state_to_huggingface': False,
        'resume_from_huggingface': False,
        'async_upload': False,
        'save_precision': 'fp16',
        'save_every_n_epochs': save_every_epoches,
        'save_every_n_steps': None,
        'save_n_epoch_ratio': None,
        'save_last_n_epochs': None,
        'save_last_n_epochs_state': None,
        'save_last_n_steps': None,
        'save_last_n_steps_state': None,
        'save_state': False,
        'resume': None,
        'train_batch_size': train_batch_size,
        'max_token_length': 225,
        'mem_eff_attn': False,
        'xformers': True,
        'max_train_steps': max_train_steps,
        'max_train_epochs': None,
        'max_data_loader_n_workers': 0,
        'persistent_data_loader_workers': False,
        'seed': seed,
        'gradient_checkpointing': False,
        'gradient_accumulation_steps': 1,
        'mixed_precision': 'fp16',
        'full_fp16': False,
        'clip_skip': clip_skip,
        'logging_dir': None,
        'log_with': None,
        'log_prefix': None,
        'log_tracker_name': None,
        'wandb_api_key': None,
        'noise_offset': 0.1,
        'multires_noise_iterations': None,
        'multires_noise_discount': 0.3,
        'adaptive_noise_scale': adaptive_noise_scale,
        'lowram': False,
        'sample_every_n_steps': None,
        'sample_every_n_epochs': None,
        'sample_prompts': None,
        'sample_sampler': sample_sampler,
        'config_file': None,
        'output_config': False,
        'prior_loss_weight': prior_loss_weight,
        'save_model_as': "safetensors",
        'optimizer_type': optimizer_type,
        'use_8bit_adam': use_8bit_adam,
        'use_lion_optimizer': False,
        'learning_rate': learning_rate,
        'max_grad_norm': 1.0,
        'optimizer_args': None,
        'lr_scheduler_type': '',
        'lr_scheduler_args': None,
        'lr_scheduler': lr_scheduler,
        'lr_warmup_steps': lr_warmup_steps,
        'lr_scheduler_num_cycles': 1,
        'lr_scheduler_power': 1,
        'dataset_config': None,
        'min_snr_gamma': 1.0,
        'scale_v_pred_loss_like_noise_pred': False,
        'weighted_captions': True,
        'no_token_padding': False,
        'stop_text_encoder_training': None,
        'network_dim': network_alpha,
        'unet_lr': 0.0001,
        'conv_alpha': 32,
        'text_encoder_lr': 5e-05,
        'algo': 'lora',
        'network_module': network_module,
        'network_args': network_args,
        'full_bf16': True,
        'no_half_vae': True,
        'in_json': None,
        'dataset_class': None,
        'vae': None,
        'base_weights': None,
        'dim_from_weights': False,
        'network_dropout': None,
        'scale_weight_norms': None,
        'network_train_text_encoder_only': None,
        'network_train_unet_only': None,
        'network_weights': None,
        'training_comment': None,
        'caption_dropout_every_n_epochs': 0,
        'caption_tag_dropout_rate': 0.0,
        'no_metadata': None,
    }
    return args