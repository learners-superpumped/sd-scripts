import requests
import json
import time
from replicate_request import training_locon

# url = 'http://localhost:5000/predictions'

# start = time.time()
# response = requests.post(url, json={
#     "input":{
#         "instance_data": "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/sample1.zip", 
#         "class_data": "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip",
#         "style_data":"https://storage.googleapis.com/stable-diffusion-server-dev/dataset/1024_inpaint_merged.zip",
#         "model_id": "test_sdxl",
#         "ckpt_base": "models/sd_xl_base_1.0.safetensors",
#         "resolution":1024,
#         "lr_scheduler": "cosine",
#         "unet_lr": 4e-4,
#         "text_lr": 4e-4,
#         "lora_dim": 8,
#         "min_snr_gamma": 1,
#         "max_train_steps": 2000,
#         "train_batch_size": 4,
#         "output_dir": "output/locon-leosam-1024-8-4e-4-reg-2000-2-1",
#         "optimizer":"Adafactor",
#         "class_token":"<hoge> person",
#         "reg_token":"person",
#         "sty_token":"<s1>",
#         "num_repeat_ins":39,
#         "num_repeat_sty":1,
#         "network_module":"lycoris.kohya",
#         "network_algo":"locon",
#         "extra": "--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
#         #"lambda_arc": 0.1,
# }})
# print(response)
# print(time.time() - start)

url = 'https://replicate.com/learners-superpumped/user-test-arcface-loss' # ??

start = time.time()
response = training_locon(
    user_id="testuser",
    version="2c62cd47aa6c2200cc4f2898063081c8c212f334bbdec5e323abff5f3dad51f0",
    instance_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/sample1-augmented.zip", 
    style_data="https://storage.googleapis.com/stable-diffusion-server-dev/dataset/1024_inpaint_merged.zip",
    class_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip",
    model_id= "test_sdxl",
    ckpt_base= "models/sd_xl_base_1.0.safetensors",
    resolution= 1024,
    lr_scheduler= "linear",
    unet_lr= 4e-4,
    text_lr= 4e-4,
    lora_dim= 4,
    min_snr_gamma= 1,
    max_train_steps= 2000,
    train_batch_size= 4,
    output_dir= "output/locon-base-sample1-augmented-2000-1024-reg-concept",
    optimizer="Adafactor",
    class_token="<hoge> person",
    sty_token="<s1> person",
    reg_token="person",
    num_repeat_ins=24,
    num_repeat_sty=1,
    network_module="lycoris.kohya",
    network_algo="locon",
    extra='''--optimizer_args scale_parameter=False relative_step=False warmup_init=False \\
    --network_train_unet_only
    
        '''
)
print(response)
print(time.time() - start)


start = time.time()
response = training_locon(
    user_id="testuser",
    version="2c62cd47aa6c2200cc4f2898063081c8c212f334bbdec5e323abff5f3dad51f0",
    instance_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/sample2-augmented.zip", 
    style_data="https://storage.googleapis.com/stable-diffusion-server-dev/dataset/1024_inpaint_merged.zip",
    class_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip",
    model_id= "test_sdxl",
    ckpt_base= "models/sd_xl_base_1.0.safetensors",
    resolution= 1024,
    lr_scheduler= "linear",
    unet_lr= 4e-4,
    text_lr= 4e-4,
    lora_dim= 4,
    min_snr_gamma= 1,
    max_train_steps= 2000,
    train_batch_size= 4,
    output_dir= "output/locon-base-sample1-augmented-2000-1024-reg-concept",
    optimizer="Adafactor",
    class_token="<hoge> person",
    sty_token="<s1> person",
    reg_token="person",
    num_repeat_ins=19,
    num_repeat_sty=1,
    network_module="lycoris.kohya",
    network_algo="locon",
    extra='''--optimizer_args scale_parameter=False relative_step=False warmup_init=False \\
    --network_train_unet_only
    
        '''
)
print(response)
print(time.time() - start)

#https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip