import os
import gc
import mimetypes
import shutil
import tarfile
import subprocess
from subprocess import call, check_call
#check_call("cd trainserver && bash makingretinaface.sh", shell=True)
import tempfile
from zipfile import ZipFile
from argparse import Namespace
import time
import torch
import json
import runpod

import sys
import os
from zipfile import ZipFile
sys.path.append("./library")

import logging
import traceback
from services.googlestorage import upload_file

logger = logging.getLogger(__name__)



def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        import sys

        sys.exit(1)


def download_dataset(instance_data, style_data, class_data):
    #print(str(instance_data), str(class_data))
    if str(instance_data).startswith('http'):
        extract_file_name = str(instance_data).split("/")[-1]
        if not os.path.isfile(extract_file_name):
            os.system(f"wget {str(instance_data)}")
    else:
        extract_file_name = instance_data
    after_extract_file_name = str(extract_file_name)[:-4]

    # extract zip contents, flattening any paths present within it
    with ZipFile(extract_file_name, "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                "__MACOSX"
            ):
                continue
            mt = mimetypes.guess_type(zip_info.filename)
            if mt and mt[0] and mt[0].startswith("image/"):
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, after_extract_file_name)
    os.remove(extract_file_name)

    if style_data is not None and str(style_data).startswith('http'):
        extract_sty_file_name = str(style_data).split("/")[-1]
        print(extract_sty_file_name)
        if not os.path.isfile(extract_sty_file_name):
            os.system(f"wget {str(style_data)}")
    elif style_data is not None:
        extract_sty_file_name = style_data

    print(style_data)
    if style_data is not None and str(style_data).endswith('.zip'):
        after_extract_sty_file_name = str(extract_sty_file_name)[:-4]
        os.system(f"unzip {extract_sty_file_name} -d {after_extract_sty_file_name} -o")
        # with ZipFile(extract_sty_file_name, "r") as zip_ref:
        #     for zip_info in zip_ref.infolist():
        #         if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
        #             "__MACOSX"
        #         ):
        #             continue
        #         # mt = mimetypes.guess_type(zip_info.filename)
        #         # if mt and mt[0] and mt[0].startswith("image/"):
        #         zip_info.filename = os.path.basename(zip_info.filename)
        #         zip_ref.extract(zip_info, after_extract_sty_file_name)
        # os.remove(extract_sty_file_name)
    
    else:
        after_extract_sty_file_name = extract_sty_file_name

    print(class_data)
    if class_data is not None and str(class_data).startswith('http'):
        extract_reg_file_name = str(class_data).split("/")[-1]
        print(extract_reg_file_name)
        if not os.path.isfile(extract_reg_file_name):
            os.system(f"wget {str(class_data)}")
    elif class_data is not None:
        extract_reg_file_name = class_data
    after_extract_reg_file_name = str(extract_reg_file_name)[:-4]

    if class_data is not None and str(class_data).endswith('.zip'):
        after_extract_reg_file_name = str(extract_reg_file_name)[:-4]
        os.system(f"unzip {extract_reg_file_name} -d {after_extract_reg_file_name} -o")
        # with ZipFile(extract_reg_file_name, "r") as zip_ref:
        #     for zip_info in zip_ref.infolist():
        #         if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
        #             "__MACOSX"
        #         ):
        #             continue
        #         # mt = mimetypes.guess_type(zip_info.filename)
        #         # if mt and mt[0] and mt[0].startswith("image/"):
        #         zip_info.filename = os.path.basename(zip_info.filename)
        #         zip_ref.extract(zip_info, after_extract_reg_file_name)
        # os.remove(extract_reg_file_name)
    
    else:
        after_extract_reg_file_name = extract_reg_file_name
    os.system("ls")
    #os.system(f"rm -rf _MACOSX")
    return after_extract_file_name, after_extract_sty_file_name, after_extract_reg_file_name




def make_dataset_config_file(res=512, bs=4, kt=2, image_dir=".", class_token="hoge person", reg_dir=None, sty_dir = None, reg_token="person", sty_token="<s1>", num_repeat_ins=1, num_repeat_sty=1):
    filename = "_dataset_config.toml"
    f = open(filename, "w")
    f.write(
        f'[general]\nshuffle_caption = true\ncaption_extension = \'.txt\'\nkeep_tokens = 1\n\n'
    )
    f.write(
        f'[[datasets]]\nresolution = {res}\nbatch_size = {bs}\nkeep_tokens = {kt}\n\n'
    )
    f.write(
        f'\t[[datasets.subsets]]\n\timage_dir = \'{image_dir}\'\n\tclass_tokens = \'{class_token}\'\n\tnum_repeats = {num_repeat_ins}\n\n'
    )
    if sty_dir is not None:
        f.write(
            f'\t[[datasets.subsets]]\n\timage_dir = \'{sty_dir}\'\n\tclass_tokens = \'{sty_token}\'\n\tnum_repeats = {num_repeat_sty}\n\n'
        )
    if reg_dir is not None:
        f.write(
            f'\t[[datasets.subsets]]\n\timage_dir = \'{reg_dir}\'\n\tclass_tokens = \'{reg_token}\'\n\tis_reg = true\n\tkeep_tokens = 1\n\tnum_repeats = 1'
        )
    f.close()
    return filename


def handler(event):
    user_id = event['input'].get("user_id", "testuser")
    instance_data = event['input'].get("instance_data", "./")
    style_data = event['input'].get("style_data", None)
    class_data = event['input'].get("class_data", None)
    resolution = event['input'].get("resolution", 512)
    train_batch_size = event['input'].get("train_batch_size", 4)
    max_train_steps = event['input'].get("max_train_steps", 2000)
    lr_scheduler = event['input'].get("lr_scheduler", "linear")
    model_id = event['input'].get("model_id", "test_model")
    ckpt_base = event['input'].get("ckpt_base", "./models/sd_xl_base_1.0.safetensors")
    output_dir = event['input'].get("output_dir", "output/locon")
    unet_lr = event['input'].get("unet_lr", 4e-4)
    text_lr = event['input'].get("text_lr", 4e-4)
    lora_dim = event['input'].get("lora_dim", 4)
    class_token = event['input'].get("class_token", "<hoge> person")
    reg_token = event['input'].get("reg_token", "person")
    sty_token = event['input'].get("sty_token", "<s1> person")
    optimizer = event['input'].get("optimizer", "Adafactor")
    num_repeat_ins = event['input'].get("num_repeat_ins", 24)
    num_repeat_sty = event['input'].get("num_repeat_sty", 1)
    extra = event['input'].get("extra", "")
    prior_loss_w = event['input'].get("prior_loss_w", 1.0)
    network_module = event['input'].get("network_module", None)
    network_algo = event['input'].get("network_algo", "locon")

    start = time.time()
    print(instance_data, style_data, class_data)
    # event = {
    #     "input": {
    #         "instance_data": instance_data,
    #         "reg_data": reg_data,
    #         "class_data": class_data,
    #         "model_id": model_id,
    #         "resolution": resolution,
    #         "train_batch_size": train_batch_size,
    #         "max_train_steps": max_train_steps,
    #         "learning_rate": learning_rate,
    #         "lr_scheduler": lr_scheduler,
    #         "lr_warmup_steps": lr_warmup_steps,
    #         "model_id": model_id,
    #     }
    # }

    # prepare model, dataset, config file.
    os.system("ls")
    instance_data, style_data, class_data = download_dataset(instance_data, style_data, class_data)
    
    if str(ckpt_base) != "models/sd_xl_base_1.0.safetensors" and str(ckpt_base) != "models/leosam.safetensors":
        ckpt_name = str(ckpt_base).split("/")[-1]
        os.system(f"cd models && wget {str(ckpt_base)}")
        os.system(f"mv models/{ckpt_name} models/mymodel.safetensors")
        ckpt_base = "models//mymodel.safetensors"
    # Another base model : later!
    # elif str(pretrained).startswith('http'):
    #     pretrained_fname = str(pretrained).split("/")[-1]
    #     url = str(pretrained)
    print(instance_data, class_data)
    print(os.path.isdir(instance_data), os.path.isdir(class_data))

    # if str(instance_data).startswith("https"):
    #     instance_data_name = str(instance_data).split("/")[-1]
    #     os.system(f"wget {str(instance_data)}")
    #     os.system(f"unzip ")

    


    dataset_config = make_dataset_config_file(res=resolution, bs=train_batch_size, image_dir=instance_data, sty_dir=style_data, reg_dir=class_data, class_token=class_token, sty_token=sty_token, reg_token=reg_token, num_repeat_ins=num_repeat_ins, num_repeat_sty=num_repeat_sty)


    command = f'''accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \\
        --pretrained_model_name_or_path={ckpt_base} \\
        --dataset_config={dataset_config} \\
        --output_dir={output_dir} \\
        --output_name={model_id} \\
        --max_train_steps={max_train_steps} \\
        --save_model_as=safetensors \\
        --prior_loss_weight={prior_loss_w} \\
        --learning_rate={unet_lr} \\
        --lr_scheduler={lr_scheduler} \\
        --optimizer_type=\"{optimizer}\" \\
        --xformers \\
        --mixed_precision=\"fp16\" \\
        --text_encoder_lr={text_lr} \\
        --gradient_checkpointing \\
        --save_every_n_epochs=100 \\
        --sample_every_n_epochs=100 \\
        --sample_prompts=\"sample_prompt.json\" \\
        --no_half_vae \\
        --network_dim={lora_dim} \\
        --network_args \"conv_dim={lora_dim}\" \"conv_alpha=1.0\" \"algo={network_algo}\" \\
        --network_module={network_module} \\
        --cache_latents \\
        --cache_latents_to_disk \\
        {extra}
        
    '''

    try:
        os.system("df -h")
        os.system(command)
    except:
        error_message: str = traceback.format_exc()
        logger.error(f"Unexpected error {error_message}")
        payload = {
            "errorMessage": error_message,
            "data": event,
            "status": "failed", # TODO: If train failed, status is failed.
            "lora_url": "",
            "refresh_worker": True,
            "outputs": [], # TODO: Check this field is needed
        }
        return payload
        #raise Exception("Train failed!")
    

    out_path = output_dir+f"/{model_id}.safetensors"

    gc.collect()
    torch.cuda.empty_cache()
    train_time = time.time() - start
    print(f"Elapsed time: {train_time} seconds")

    print(subprocess.check_output("/usr/bin/date"))

    print(out_path)
    upload_lora_file_path = out_path

    logger.warn(f"upload file path: {upload_lora_file_path}")
    upload_start = time.time()
    lora_uri = ""
    lora_uri = upload_file(upload_lora_file_path, directory=user_id, content_type="text/plain", use_random=model_id == 'testmodel') # TODO: Async

    upload_time = time.time() - upload_start
    # Send Webhook
    webhook_url = event["input"].get("webhook", "")
    payload = {
        "data": event,
        "input": event["input"],
        "status": "succeeded", # TODO: If train failed, status is failed.
        "lora_url": lora_uri,
        "refresh_worker": True,
        "metrics": {
            "train_time": train_time,
            "upload_time": upload_time,
        },
        "outputs": [], # TODO: Check this field is needed
    }
    logger.warn(f"Success webhook called {lora_uri}")
    
    return payload
    
runpod.serverless.start({
    "handler": handler
})

# if __name__=='__main__':
#     with open("test_input.json", "r") as f:
#         json_data = json.load(f)
#     handler(json_data)
