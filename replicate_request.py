import requests
import json
import time
from typing import Optional



#@retry(stop_max_attempt_number=5)
def training_locon(
    user_id: str, 
    version: str, 
    instance_data: str,
    class_data: str,
    model_id: str,
    ckpt_base: str,
    resolution: int,
    lr_scheduler: str,
    unet_lr: float,
    text_lr: float,
    lora_dim: int,
    min_snr_gamma: int,
    max_train_steps: int,
    train_batch_size: int,
    output_dir: str,
    optimizer: str,
    class_token: str,
    reg_token: str,
    extra: str,
    num_repeat: int,
):
    url = 'https://api.replicate.com/v1/predictions'
    data = {
        "version": version,
        "input":{
            "instance_data": instance_data,
            "class_data": class_data,
            "model_id": model_id,
            "ckpt_base": ckpt_base,
            "resolution": resolution,
            "lr_scheduler": lr_scheduler,
            "unet_lr": unet_lr,
            "text_lr": text_lr,
            "min_snr_gamma": min_snr_gamma,
            "max_train_steps": max_train_steps,
            "train_batch_size": train_batch_size,
            "lora_dim": lora_dim,
            "output_dir": output_dir,
            "extra": extra,
            "optimizer": optimizer,
            "class_token": class_token,
            "reg_token": reg_token,
            "num_repeat": num_repeat,
        }
    }

    REPLICATE_HEADER = {
        'Authorization': 'Token r8_bksyivoSMFuaFWDCaydMz1dA6mlmYL505f3mS',
    }
    response = requests.post(
        url, 
        json=data,
        headers=REPLICATE_HEADER
    )
    print(response)
    response_json = response.json()
    # if not response_json.get("id"):
    #     if response_json.get("retry_after"):
    #         time.sleep(response_json.get("retry_after") + 3)
    #     raise ValueError(response_json)
    return response_json


