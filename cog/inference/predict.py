import os
from typing import Iterator
from time import time

import torch
from cog import BasePredictor, Input, Path
from weights import WeightsDownloadCache
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from services.locon_inference_service import LoconInferenceService
from schema.inference_request import IMG2IMGInferenceDTO, DDSDInferenceDTO, CoupleInferenceDTO, TXT2IMGInferenceDTO


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        checkpoint_path = "models/sd_xl_base_1.0.safetensors" 
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            local_file_only=True,
            torch_dtype=torch.float16,
        ).to(self.device, torch.float16)
        self.txt2img_pipe = StableDiffusionXLPipeline(
            vae=self.img2img_pipe.vae,
            text_encoder=self.img2img_pipe.text_encoder,
            text_encoder_2=self.img2img_pipe.text_encoder_2,
            tokenizer=self.img2img_pipe.tokenizer,
            tokenizer_2=self.img2img_pipe.tokenizer_2,
            unet=self.img2img_pipe.unet,
            scheduler=self.img2img_pipe.scheduler,
        ).to(self.device, torch.float16)

        self.weights_download_cache = WeightsDownloadCache(2**30)

    def get_locon(self, url: str):
        if 'replicate.delivery' in url:
            url = url.replace('replicate.delivery/pbxt', 'storage.googleapis.com/replicate-files')

        path = self.weights_download_cache.ensure(url)
        return path

    @torch.inference_mode()
    def predict(
        self,
        locon_url: str = Input(
            description="URL of LoCon to use",
            default=None,
        ),
        image: Path = Input(
            description="Optional Image to use for img2img guidance", default=None
        ),
        prompt: str = Input(
            description="Input prompt",
            default="leogirl, hoge person, realistic Documentary photography, detailed face cleavage, realistic, photorealistic",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="(worst quality, low quality, cgi, bad eye, worst eye, illustration, cartoon), deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, open mouth",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=10,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.5,
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=True
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=1234
        ),
        type: str = Input(
            description="Type of inference", default="ddsd"
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        # 시간 절감
        # os.system("nvidia-smi")

        # print('##### DISK #####')
        # os.system("df -h")
        # os.system("free -h")
        # print(self.weights_download_cache.cache_info())

        if image:
            print("Using img2img pipeline")
            pipe = self.img2img_pipe
            extra_kwargs = {}
        else:
            print("Using txt2img pipeline")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        locon_path = self.get_locon(locon_url)

        dto_input = {
            "image_path": str(image),
            "lora_path": locon_path,
            "params": [
                {
                    "guidance_scale": guidance_scale,
                    "prompt_strength": prompt_strength,
                    "num_inference_steps": num_inference_steps,
                }
            ],
            "disable_safety_check": disable_safety_check,
            "num_outputs": num_outputs,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": "KarrasDPM",
            "width": 1024,
            "height": 1024,
            "type": "ddsd",
            "seed": seed
        }
        if type == "ddsd":
            inference_dto = DDSDInferenceDTO(**dto_input)
        elif type == "couple":
            inference_dto = CoupleInferenceDTO(**dto_input)
        elif type == "txt2img":
            inference_dto = TXT2IMGInferenceDTO(**dto_input)
        else:
            inference_dto = IMG2IMGInferenceDTO(**dto_input)
        locon_inference_service = LoconInferenceService()
        generated_images = locon_inference_service.predict(inference_dto)

        result_count = 0
        for i, img in enumerate(generated_images):
            output_path = f"/tmp/seed-{seed}-{i}.png"
            img.save(output_path)
            yield Path(output_path)
            result_count += 1

        if result_count == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )


def make_scheduler():
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
    return scheduler
