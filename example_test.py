import requests
import json
import time
from replicate_request import training_locon

url = 'http://localhost:5000/predictions'

# start = time.time()
# response = requests.post(url, json={
#     "input":{
#         "instance_data": "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip", 
#         "class_data": "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip",
#         "model_id": "test_sdxl",
#         "ckpt_base": "models/leosam.safetensors",
#         "resolution": 512,
#         "lr_scheduler": "cosine",
#         "unet_lr": 2e-4,
#         "text_lr": 1e-5,
#         "lora_dim": 32,
#         "min_snr_gamma": 1,
#         "max_train_steps": 10,
#         "train_batch_size": 1,
#         "output_dir": "output/locon",
#         "optimizer": "Adafactor",
#         "extra": "--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
#         #"lambda_arc": 0.1,
# }})
# print(response)
# print(time.time() - start)

url = 'https://replicate.com/learners-superpumped/user-test-arcface-loss' # ??

start = time.time()
response = training_locon(
    user_id="testuser",
    version="273bf581f559eec1e07909bcf75d6dde79509c2748e94c194c49a76d85f5f082",
    instance_data= "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip", 
    class_data= "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip",
    model_id= "test_sdxl",
    ckpt_base= "models/leosam.safetensors",
    resolution= 512,
    lr_scheduler= "linear",
    unet_lr= 4e-4,
    text_lr= 4e-4,
    lora_dim= 32,
    min_snr_gamma= 1,
    max_train_steps= 1,
    train_batch_size= 1,
    output_dir= "output/locon",
    optimizer="Adafactor",
    extra="--optimizer_args scale_parameter=False relative_step=False warmup_init=False"
)
print(response)
print(time.time() - start)