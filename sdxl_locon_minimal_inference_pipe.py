from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from safetensors import safe_open
from safetensors.torch import load_file
import time
import json
from lycoris.kohya import LycorisNetwork
from lycoris.config import PRESET
from lycoris.utils import get_module
from lycoris.modules import make_module
import os
import random
import numpy as np
from tqdm import tqdm
import math
from einops import repeat
import datetime
from PIL import Image
from diffusers import EulerDiscreteScheduler
from sdxl_minimal_inference import get_timestep_embedding
# Tokenizers
from library import sdxl_train_util, sdxl_model_util
from transformers import CLIPTokenizer
import open_clip
import argparse
from diffusers.utils import load_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--num_images_per_prompt", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", type=str, default="inference_results/img2img")
    parser.add_argument("--concept_image", type=str, default="image/2.jpg")
    parser.add_argument("--locon_model", type=str, default="output/locon-augmented-kr-leosam-512-32-3e-4-reg-small-1500")
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--target_width", type=int, default=1024)
    parser.add_argument("--target_height", type=int, default=1024)
    parser.add_argument("--prompt", type=str, default="leogirl, <hoge> person, detailed face cleavage, realistic, photorealistic")
    parser.add_argument("--negative_prompt", type=str, default= "(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth")
    parser.add_argument("--algo", type=str, default="locon")
    parser.add_argument("--mode", type=str, default="i2i")
    parser.add_argument("--ckpt_path", type=str, default="models/leosam.safetensors")
    return parser.parse_args()


# load SDXL pipeline
if __name__ == "__main__":
    args = parse_args()
    base_pipe = StableDiffusionXLPipeline.from_single_file(
        args.ckpt_path,
        local_file_only=True,
        torch_dtype=torch.float16,
    )
    base_pipe.load_lora_weights(args.locon_model)
    if args.mode == "i2i":
        pipe = StableDiffusionXLImg2ImgPipeline(
            vae=base_pipe.vae,
            unet=base_pipe.unet,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            scheduler=base_pipe.scheduler,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
        )

    elif args.mode == "t2i":
        pipe = base_pipe

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # dtype
    pipe.unet.to("cuda", torch.float16)
    pipe.vae.to("cuda", torch.float16)
    pipe.text_encoder.to("cuda", torch.float16)
    pipe.text_encoder_2.to("cuda", torch.float16)

    # scheduler
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    pipe.scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    # concept_image = Image.open(args.concept_image).convert("RGB")
    # width, height = concept_image.size
    # ratio = height / width 
    # new_width = 1024
    # new_height = int(new_width * ratio)
    # print("new width: ", new_width)
    # print("new height: ", new_height)
    # concept_image = concept_image.resize((new_width, new_height))
    generator = torch.Generator("cuda").manual_seed(args.seed)

    concept_image = load_image(args.concept_image)
    width, height = concept_image.size
    ratio = height / width 
    new_width = 1024
    new_height = int(new_width * ratio)
    print("new width: ", new_width)
    print("new height: ", new_height)
    concept_image = concept_image.resize((new_width, new_height))
    
    print(np.array(concept_image))
    if args.mode == "i2i":
        images = pipe(
            image=concept_image,
            prompt=args.prompt,
            prompt_2="", # dummy
            negative_prompt=args.negative_prompt,
            negative_prompt_2="", # dummy
            guidance_scale=args.guidance_scale,
            # width=args.target_width,
            # height=args.target_height,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            # aesthetic_score=args.ae/sthetic_score,
            strength=args.strength,
            generator=generator,
        ).images
    elif args.mode == "t2i":
        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            width=args.target_width,
            height=args.target_height,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
        ).images
        
    for i, img in enumerate(images):
        img.save(f"{args.output_dir}/{i}.png")
