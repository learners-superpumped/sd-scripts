from diffusers import StableDiffusionXLPipeline
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=int, default=5)
    parser.add_argument("--target_height", type=int, default=768)
    parser.add_argument("--target_width", type=int, default=768)
    parser.add_argument("--original_height", type=int, default=768)
    parser.add_argument("--original_width", type=int, default=768)
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_left", type=int, default=0)
    parser.add_argument("--locon_model", type=str, default="output/locon-kr-leosam-768-32-2/test.safetensors")
    parser.add_argument("--lora_strength", type=float, default=1)
    parser.add_argument("--prompt", type=str, default="leogirl, hoge girl, realistic Documentary photography, detailed face cleavage, realistic, photorealistic")
    parser.add_argument("--negative_prompt", type=str, default= "(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth")
    parser.add_argument("--algo", type=str, default="locon")

    return parser.parse_args()


def create_network_from_weights(multiplier, file, vae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    # get dim/alpha mapping
    unet_loras = {}
    te_loras = {}
    LycorisNetwork.apply_preset(PRESET["full"])

    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        
        if lora_name.startswith(LycorisNetwork.LORA_PREFIX_UNET):
            unet_loras[lora_name] = None
        elif lora_name.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
            te_loras[lora_name] = None

    for name, modules in unet.named_modules():
        lora_name = f'{LycorisNetwork.LORA_PREFIX_UNET}_{name}'.replace('.','_')
        if lora_name in unet_loras:
            unet_loras[lora_name] = modules

    for text_encoder in text_encoders:
        for name, modules in text_encoder.named_modules():
            lora_name = f'{LycorisNetwork.LORA_PREFIX_TEXT_ENCODER}_{name}'.replace('.','_')
            if lora_name in te_loras:
                te_loras[lora_name] = modules
   
    network = LycorisNetwork(text_encoders, unet)
    network.unet_loras = []
    network.text_encoder_loras = []
    
    for lora_name, orig_modules in unet_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.unet_loras.append(module)
    
    for lora_name, orig_modules in te_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.text_encoder_loras.append(module)
    
    for lora in network.unet_loras + network.text_encoder_loras:
        lora.multiplier = multiplier
    
    return network, weights_sd

# load SDXL pipeline
checkpoint_path =  "./models/leosam.safetensors"

text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, checkpoint_path, "cpu"
)

if __name__ == "__main__":
    args = parse_args()
    if args.algo == "lora":
        raise NotImplementedError("LORA is not supported yet")
    else:
        # for SDXL two text_encoders
        network, weights_sd = create_network_from_weights(
            multiplier=args.lora_strength,
            file=args.locon_model,
            vae=vae,
            text_encoders=[text_encoder1, text_encoder2],
            unet=unet,
            for_inference=True,
            algo=args.algo
        )
        network.apply_to([text_encoder1, text_encoder2], unet, True, True)
        # info = network.load_state_dict(weights_sd, False)
        network.to("cuda", dtype=torch.float16)


    DTYPE = torch.float16
    DEVICE = "cuda"
    unet.to(DEVICE, dtype=DTYPE)
    unet.eval()

    unet.set_use_memory_efficient_attention(True, False)
    if torch.__version__ >= "2.0.0": # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(True)

    vae_dtype = torch.float32
    vae.to(DEVICE, dtype=vae_dtype)
    vae.eval()

    text_encoder1.to(DEVICE, dtype=DTYPE)
    text_encoder1.eval()
    text_encoder2.to(DEVICE, dtype=DTYPE)
    text_encoder2.eval()

    # scheduler
    SCHEDULER_LINEAR_START = 0.00085
    SCHEDULER_LINEAR_END = 0.0120
    SCHEDULER_TIMESTEPS = 1000
    SCHEDLER_SCHEDULE = "scaled_linear"
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    # argument
    target_height = args.target_height
    target_width = args.target_width
    original_height = args.original_height
    original_width = args.original_width
    crop_top = args.crop_top
    crop_left = args.crop_left

    steps = args.steps
    guidance_scale = args.guidance_scale

    # HuggingFaceのmodel id
    text_encoder_1_name = "openai/clip-vit-large-patch14"
    text_encoder_2_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    tokenizer1 = CLIPTokenizer.from_pretrained(text_encoder_1_name)
    tokenizer2 = lambda x: open_clip.tokenize(x, context_length=77)


    def generate_image(text_model1, text_model2, unet, vae, prompt, prompt2, negative_prompt, seed=None):
        # 将来的にサイズ情報も変えられるようにする / Make it possible to change the size information in the future
        # prepare embedding
        with torch.no_grad():
            # vector
            emb1 = get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
            emb2 = get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
            emb3 = get_timestep_embedding(torch.FloatTensor([target_height, target_width]).unsqueeze(0), 256)
            # print("emb1", emb1.shape)
            c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(DEVICE, dtype=DTYPE)
            uc_vector = c_vector.clone().to(DEVICE, dtype=DTYPE)  # ちょっとここ正しいかどうかわからない I'm not sure if this is right

            # crossattn

        # Text Encoderを二つ呼ぶ関数  Function to call two Text Encoders
        def call_text_encoder(text, text2):
            # text encoder 1
            batch_encoding = tokenizer1(
                text,
                truncation=True,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(DEVICE)

            with torch.no_grad():
                enc_out = text_model1(tokens, output_hidden_states=True, return_dict=True)
                text_embedding1 = enc_out["hidden_states"][11]
                # text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)    # layer normは通さないらしい

            # text encoder 2
            with torch.no_grad():
                tokens = tokenizer2(text2).to(DEVICE)

                enc_out = text_model2(tokens, output_hidden_states=True, return_dict=True)
                text_embedding2_penu = enc_out["hidden_states"][-2]
                # print("hidden_states2", text_embedding2_penu.shape)
                text_embedding2_pool = enc_out["text_embeds"]   # do not support Textual Inversion

            # 連結して終了 concat and finish
            text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)
            return text_embedding, text_embedding2_pool

        # cond
        c_ctx, c_ctx_pool = call_text_encoder(prompt, prompt2)
        # print(c_ctx.shape, c_ctx_p.shape, c_vector.shape)
        c_vector = torch.cat([c_ctx_pool, c_vector], dim=1)

        # uncond
        uc_ctx, uc_ctx_pool = call_text_encoder(negative_prompt, negative_prompt)
        uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

        text_embeddings = torch.cat([uc_ctx, c_ctx])
        vector_embeddings = torch.cat([uc_vector, c_vector])

        # メモリ使用量を減らすにはここでText Encoderを削除するかCPUへ移動する

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # # random generator for initial noise
            # generator = torch.Generator(device="cuda").manual_seed(seed)
            generator = None
        else:
            generator = None

        # get the initial random noise unless the user supplied it
        # SDXLはCPUでlatentsを作成しているので一応合わせておく、Diffusersはtarget deviceでlatentsを作成している
        # SDXL creates latents in CPU, Diffusers creates latents in target device
        latents_shape = (1, 4, target_height // 8, target_width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(DEVICE, dtype=DTYPE)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        # set timesteps
        scheduler.set_timesteps(steps, DEVICE)

        # このへんはDiffusersからのコピペ
        # Copy from Diffusers
        timesteps = scheduler.timesteps.to(DEVICE)  # .to(DTYPE)
        num_latent_input = 2
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                print("i", i)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
                print("latent_model_input", torch.sum(torch.isnan(latent_model_input)))
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                print("latent_model_input", torch.sum(torch.isnan(latent_model_input)))
                print("t", t)
                print("text", torch.sum(torch.isnan(text_embeddings)))
                print("vector", torch.sum(torch.isnan(vector_embeddings)))
                noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)
                print("noise pred", torch.sum(torch.isnan(noise_pred)))
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # latents = 1 / 0.18215 * latents
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
            latents = latents.to(vae_dtype)
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # image = self.numpy_to_pil(image)
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        # 保存して終了 save and finish
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for i, img in enumerate(image):
            img.save(os.path.join("inference_results", f"image_{timestamp}_{i:03d}.png"))

    generate_image(
        text_encoder1,
        text_encoder2,
        unet,
        vae,
        args.prompt,
        "",
        args.negative_prompt,
        args.seed
    )
    print("Done!")
