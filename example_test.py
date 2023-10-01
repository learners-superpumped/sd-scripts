import requests
import json
import time
from replicate_request import training_locon

# url = 'http://localhost:5000/predictions'

# start = time.time()
# response = requests.post(url, json={
#     "input":{
#         "instance_data": "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip", 
#         "class_data": "https://storage.googleapis.com/snow_image/kor_latent/k-faces-large-flat.zip",
#         "model_id": "test_sdxl",
#         "ckpt_base": "https://civitai.com/api/download/models/150851",
#         "resolution": 512,
#         "lr_scheduler": "cosine",
#         "unet_lr": 2e-4,
#         "text_lr": 1e-5,
#         "lora_dim": 32,
#         "min_snr_gamma": 1,
#         "max_train_steps": 10,
#         "train_batch_size": 4,
#         "output_dir": "output/locon",
#         #"lambda_arc": 0.1,
# }})
# print(response)
# print(time.time() - start)

url = 'https://replicate.com/learners-superpumped/user-test-arcface-loss' # ??

start = time.time()
response = training_locon(
    user_id="testuser",
    version="3a74db1051b15dbd38f59951b04e488f19015a37ae72171c52c59d55618e3759",
    instance_data= "https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip", 
    class_data= "https://storage.googleapis.com/snow_image/kor_latent/k-faces-large-flat.zip",
    model_id= "test_sdxl",
    ckpt_base= "https://civitai.com/api/download/models/150851",
    resolution= 512,
    lr_scheduler= "cosine",
    unet_lr= 2e-4,
    text_lr= 1e-5,
    lora_dim= 32,
    min_snr_gamma= 1,
    max_train_steps= 1000,
    train_batch_size= 4,
    output_dir= "output/locon",
)
print(response)
print(time.time() - start)