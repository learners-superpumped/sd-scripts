def make_dataset_config_file(res=512, bs=4, kt=2, image_dir=".", class_token="hoge person", reg_dir=None, reg_token="person"):
    filename = "_dataset_config.toml"
    f = open(filename, "w")
    f.write(
        f'[general]\nshuffle_caption = true\ncaption_extension = \'.txt\'\nkeep_tokens = 1\n\n'
    )
    f.write(
        f'[[datasets]]\nresolution = {res}\nbatch_size = {bs}\nkeep_tokens = {kt}\n'
    )
    f.write(
        f'\t[[datasets.subsets]]\n\timage_dir = \'{image_dir}\'\n\tclass_tokens = \'{class_token}\'\n\n'
    )
    if reg_dir is not None:
        f.write(
            f'\t[[datasets.subsets]]\n\timage_dir = \'{reg_dir}\'\n\tclass_tokens = \'{reg_token}\'\n\tis_reg = true\n\tkeep_tokens = 1\n\tnum_repeats = 1'
        )
    f.close()
    return filename
pretrained = "Dfafd"
output_dir = "Dfafe"
max_train_steps = 1
unet_lr = 1.1
lr_scheduler = "adfa"
text_lr = 2.2321
lora_dim = 122
resolution = 512
bs = 4
instance_data = "12111"
class_data = "3r2eew"
train_batch_size =2

dataset_config = make_dataset_config_file(res=resolution, bs=train_batch_size, image_dir=instance_data, reg_dir=class_data)
command = f'''
    accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \\ 
    --pretrained_model_name_or_path={pretrained} \\ 
    --dataset_config={dataset_config} \\ 
    --output_dir={output_dir} \\ 
    --output_name=test \\ 
    --max_train_steps={max_train_steps} \\ 
    --save_model_as=safetensors \\ 
    --prior_loss_weight=1.0 \\ 
    --learning_rate={unet_lr} \\ 
    --lr_scheduler={lr_scheduler} \\ 
    --optimizer_type=\"AdamW8bit\" \\ 
    --xformers \\ 
    --mixed_precision=\"fp16\" \\ 
    --cache_latents \\ 
    --text_encoder_lr={text_lr} \\ 
    --gradient_checkpointing \\ 
    --save_every_n_epochs=100 \\ 
    --sample_every_n_epochs=100 \\ 
    --sample_prompts=\"sample_prompt.json\" \\ 
    --no_half_vae \\ 
    --network_dim={lora_dim} \\ 
    --network_args \"conv_dim={lora_dim}\" \"conv_alpha=1.0\" \"algo=locon\" \\ 
    --network_module=lycoris.kohya \\ 
'''
print(command)