import os
import shutil
from typing import Iterator
from time import time

import torch
from cog import BasePredictor, Input, Path
from weights import WeightsDownloadCache
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

import settings


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        checkpoint_path = settings.BASE_MODEL_PATH 
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
            safety_checker=self.img2img_pipe.safety_checker,
            feature_extractor=self.img2img_pipe.feature_extractor,
        ).to(self.device, torch.float16)

        self.weights_download_cache = WeightsDownloadCache(2**30)

    def get_locon(self, pipe, url: str):
        if 'replicate.delivery' in url:
            url = url.replace('replicate.delivery/pbxt', 'storage.googleapis.com/replicate-files')

        path = self.weights_download_cache.ensure(url)
        return self.gpu_weights(pipe, path)

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
        mask: Path = Input(
            description="Optional Mask to use for legacy inpainting", default=None
        ),
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        prompt_2: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        negative_prompt_2: str = Input(
            description="Specify things to not see in the output",
            default=None,
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
        scheduler: str = Input(
            default="KarrasDPM",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
                "KarrasDPM",
            ],
            description="Choose a scheduler.",
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
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
            image = load_image(str(image))
        if mask:
            mask = load_image(str(mask))

        if image:
            print("Using img2img pipeline")
            pipe = self.img2img_pipe
        else:
            print("Using txt2img pipeline")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")


        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = make_scheduler()
        locon_start = time()
        self.get_locon(pipe, locon_url)
        print(f"LoCon loading time: {(time() - locon_start):.2f}")

        if disable_safety_check:
            pipe.safety_checker = None
        else:
            pipe.safety_checker = self.safety_checker

        result_count = 0
    
        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_inference_steps,
            strength=prompt_strength,
            generator=generator,
            **extra_kwargs,
        )

        for i, img in enumerate(output.images):
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
