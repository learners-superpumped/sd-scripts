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
#         "model_id": "test_sdxl",
#         "ckpt_base": "models/leosam.safetensors",
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
#         "num_repeat":95,
#         "extra": "--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
#         #"lambda_arc": 0.1,
# }})
# print(response)
# print(time.time() - start)

url = 'https://replicate.com/learners-superpumped/user-test-arcface-loss' # ??

start = time.time()
response = training_locon(
    user_id="testuser",
    version="ecda5fdf9e1cfff66e6988ca1991ca1630c38c7208d241415d6d9360f7cf141a",
    instance_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/augmented_sample2.zip", 
    class_data= "https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip",
    model_id= "test_sdxl",
    ckpt_base= "models/leosam.safetensors",
    resolution= 1024,
    lr_scheduler= "linear",
    unet_lr= 4e-4,
    text_lr= 1e-5,
    lora_dim= 8,
    min_snr_gamma= 1,
    max_train_steps= 2000,
    train_batch_size= 4,
    output_dir= "output/locon-leosam-1024-8-4e-4-1e-5-reg-2000-2-2-aug",
    optimizer="Adafactor",
    class_token="<hoge> person",
    reg_token="person",
    num_repeat=39,
    extra="--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
)
print(response)
print(time.time() - start)


#https://storage.googleapis.com/stable-diffusion-server-dev/dataset/k-faces-small-flat.zip