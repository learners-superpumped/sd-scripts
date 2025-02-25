accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
    --pretrained_model_name_or_path=models/sd_xl_base_1.0.safetensors \
    --dataset_config=_typos.toml \
    --output_dir=output/text-encoder-kr2 \
    --output_name=test \
    --max_train_steps=1000 \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --learning_rate=4e-4 \
    --lr_scheduler="cosine" \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --text_encoder_lr=4e-4 \
    --gradient_checkpointing \
    --save_every_n_steps=500 \
    --sample_every_n_steps=200 \
    --sample_sampler=euler_a \
    --sample_prompts="sample_prompt.json"\
    --no_half_vae \
    --network_dim=8 \
    --network_args "conv_dim=8" "conv_alpha=1.0" "algo=locon"\
    --network_module=lycoris.kohya
