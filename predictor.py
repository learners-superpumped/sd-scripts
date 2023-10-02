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

from cog import BaseModel, BasePredictor, Input, Path
import sys
import os
from zipfile import ZipFile
sys.path.append("./library")



class TrainingOutput(BaseModel):
    weights: Path

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        import sys

        sys.exit(1)


def download_dataset(instance_data, class_data):
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


    if class_data is not None and str(class_data).startswith('http'):
        extract_reg_file_name = str(class_data).split("/")[-1]
        print(extract_reg_file_name)
        if not os.path.isfile(extract_reg_file_name):
            os.system(f"wget {str(class_data)}")
    elif class_data is not None:
        extract_reg_file_name = class_data
    after_extract_reg_file_name = str(extract_reg_file_name)[:-4]

    if class_data is not None:
        with ZipFile(extract_reg_file_name, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, after_extract_reg_file_name)
        os.remove(extract_reg_file_name)

    return after_extract_file_name, after_extract_reg_file_name




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

class Predictor(BasePredictor):
    def setup(self):
        # HACK: wait a little bit for instance to be ready
        time.sleep(5)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
        self,
        instance_data: str = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
            default="./"
        ),
        class_data: str = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=2,
        ),
        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=1000,
        ),
        lr_scheduler: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant_with_warmup",
        ),
        model_id: str = Input(
            description="model save path",
            default="test_model",
        ),
        lr_warmup_steps: int = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=176,
        ),
        ckpt_base: str = Input(
            description="pretrained model name of path",
            default="https://civitai.com/api/download/models/150851",
        ),
        output_dir: str = Input(
            description="output dir of finetuned model",
            default="output/locon",
        ),
        unet_lr: float = Input(
            description="lr of unet",
            default=1e-4
        ),
        text_lr: float = Input(
            description="lr of text encoder",
            default=1e-5
        ),
        lora_dim: int = Input(
            description="dim of lora, locon",
            default=32
        ),
        class_token: str = Input(
            description="learnable token",
            default="hoge girl"
        ),
        reg_token: str = Input(
            description="regularization token",
            default="girl"
        ),
        optimizer: str = Input(
            description="optimizer will be used",
            default="AdamW8bit"
        ),
        extra: str = Input(
            description="extra args: for your need, for example, optimizer_args, ...",
            default="AdamW8bit"
        )
    ) -> TrainingOutput:
        start = time.time()
        print(instance_data, class_data)
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
        instance_data, class_data = download_dataset(instance_data, class_data)
        
        # if str(ckpt_base) == "https://civitai.com/api/download/models/150851":
        #     os.system("make leosam")
        #     ckpt_base = "models/leosam.safetensors"
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

        


        dataset_config = make_dataset_config_file(res=resolution, bs=train_batch_size, image_dir=instance_data, reg_dir=class_data, class_token=class_token, reg_token=reg_token)


        command = f'''accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \\
            --pretrained_model_name_or_path={ckpt_base} \\
            --dataset_config={dataset_config} \\
            --output_dir={output_dir} \\
            --output_name={model_id} \\
            --max_train_steps={max_train_steps} \\
            --save_model_as=safetensors \\
            --prior_loss_weight=1.0 \\
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
            --network_args \"conv_dim={lora_dim}\" \"conv_alpha=1.0\" \"algo=locon\" \\
            --network_module=lycoris.kohya \\
            --cache_latents \\
            --cache_latents_to_disk \\
            {extra}
            
        '''
        # ff = open("./train.sh", "w")
        # ff.write(command)
        # ff.close()
        try:
            os.system("df -h")
            os.system(command)
        except:
            raise Exception("Train failed!")
        

        out_path = output_dir+f"/{model_id}.safetensors"

        gc.collect()
        torch.cuda.empty_cache()
        call("nvidia-smi")

        print(f"Elapsed time: {time.time() - start} seconds")

        print(subprocess.check_output("/usr/bin/date"))

        print(out_path)
        print(Path(out_path))
        return TrainingOutput(weights=Path(out_path))

if __name__=='__main__':
    p = Predictor()
    p.predict(instance_data="https://storage.googleapis.com/peekaboo-studio/userinputzip/b6ntuv40b6result.zip",
              class_data="https://storage.googleapis.com/snow_image/kor_latent/k-faces-large-flat.zip")
