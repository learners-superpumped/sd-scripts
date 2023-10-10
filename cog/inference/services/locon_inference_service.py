import torch
import os
from typing import List, Tuple

from PIL import Image, ImageOps
from skimage import exposure
import cv2
from services.segmentation import Segmentator

import numpy as np
import mediapipe as mp

from diffusers import StableDiffusionImg2ImgPipeline
from compel import Compel
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from RealESRGAN import RealESRGAN

from schema.inference_request import (
    IMG2IMGInferenceDTO, DDSDInferenceDTO, CoupleInferenceDTO, TXT2IMGInferenceDTO
)
from utils.create_scheduler import make_scheduler
from utils.config import conf
import logging
from services.abstract_inference_service import InferenceService

from utils.config import conf
from utils.const import BACKGROUND, HAIR, FACE, BODY, CLOTHES, OTHERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


class LoconInferenceService(InferenceService):
    def __init__(self):
        self.txt2img_pipe, self.img2img_pipe, self.inpainting_pipe = self._create_locon_pipelines()
        self.inpainting_pipe.safety_checker = None
        self.compel = Compel(
            tokenizer=self.img2img_pipe.tokenizer,
            text_encoder=self.img2img_pipe.text_encoder,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Upscaler: Cropped face
        self.upscaler = RealESRGAN(device, scale=4)
        self.upscaler.load_weights(f'{conf.BASE_MODEL_PATH}/RealESRGAN_x4.pth', download=True)

        # Segmentator
        self.segmentator = Segmentator()

        self.inpainting_pipe.safety_checker = None
    
    def _create_locon_pipelines(self) -> StableDiffusionImg2ImgPipeline:
        txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
            os.path.join(conf.BASE_MODEL_PATH, conf.BASE_MODEL_NAME),
            torch_dtype=torch.float16,
            cache_dir=conf.MODEL_CACHE,
            local_files_only=True,
        ).to(conf.DEVICE)

        img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
            unet=txt2img_pipe.unet,
            scheduler=txt2img_pipe.scheduler,
        )
        inpainting_pipe = StableDiffusionXLInpaintPipeline(
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
            unet=txt2img_pipe.unet,
            scheduler=txt2img_pipe.scheduler,
        )
        return txt2img_pipe, img2img_pipe, inpainting_pipe

    def predict(self, locon_inference_dto: IMG2IMGInferenceDTO) -> List[Image.Image]:
        if isinstance(locon_inference_dto, DDSDInferenceDTO):
            return self._inpainting(locon_inference_dto)
        elif isinstance(locon_inference_dto, CoupleInferenceDTO):
            return self._couple(locon_inference_dto)
        elif isinstance(locon_inference_dto, IMG2IMGInferenceDTO):
            return self._i2i(locon_inference_dto)
        elif isinstance(locon_inference_dto, TXT2IMGInferenceDTO):
            return self._t2i(locon_inference_dto)
        else:
            raise ValueError("Invalid Inference DTO")

    def _inpaint_face(
        self,
        locon_inference_dto: DDSDInferenceDTO,
        images: List[Image.Image],
        bbox: Tuple[int, int, int, int], # (left, top, right, bottom)
        kernel_size: Tuple[int, int],
        center_coord_rate: Tuple[float, float] = None,
    ) -> List[Image.Image]:
        
        upscaled_cropped_faces = []
        resized_blurred_face_masks = []
        brightnesses = []
        for image in images:
            cropped_face = image.crop(bbox)
            blurred_face_mask, brightness = self._extract_blurred_face_mask(
                cropped_face,
                kernel_size,
                center_coord_rate=center_coord_rate
            )
            brightnesses.append(brightness)

            # matplotlib save fig
            import matplotlib.pyplot as plt
            plt.imshow(cropped_face)
            plt.imshow(blurred_face_mask, alpha=0.5)
            plt.savefig("save_dir/blurred_face_mask.png")

            # Upscale cropped face
            upscaled_cropped_face = self.upscaler.predict(cropped_face)

            # Reisze
            upscaled_cropped_face = upscaled_cropped_face.resize(
                (locon_inference_dto.width, locon_inference_dto.height)
            )
            upscaled_cropped_faces.append(upscaled_cropped_face)
            resized_blurred_face_mask = cv2.resize(
                blurred_face_mask,
                (locon_inference_dto.width, locon_inference_dto.height)
            )
            resized_blurred_face_masks.append(resized_blurred_face_mask)

        self.inpainting_pipe.scheduler = make_scheduler(
            locon_inference_dto.scheduler, self.inpainting_pipe.scheduler.config
        )
        logger.debug("Loaded Scheduler")

        if locon_inference_dto.disable_safety_check:
            self.img2img_pipe.safety_checker = None
        
        generator = torch.Generator(conf.DEVICE)
        logger.debug("Loaded Prompt Embeds")

        param = locon_inference_dto.params[0]
        generator.manual_seed(locon_inference_dto.seed)

        generated_image = self.inpainting_pipe(
            prompt=[locon_inference_dto.prompt] * locon_inference_dto.num_outputs,
            negative_prompt=[locon_inference_dto.negative_prompt] * locon_inference_dto.num_outputs,
            num_inference_steps=param.num_inference_steps,
            guidance_scale=param.guidance_scale,
            generator=generator,
            strength=param.prompt_strength,
            image=upscaled_cropped_faces,
            mask_image=resized_blurred_face_masks,
            width=locon_inference_dto.width,
            height=locon_inference_dto.height,
        )
        
        # Smooth original image by blurred_face_mask 
        crop_left, crop_top, crop_right, crop_bottom = bbox
        blurred_face_mask = cv2.resize(blurred_face_mask, (crop_right - crop_left, crop_bottom - crop_top))
        blurred_face_mask = np.expand_dims(blurred_face_mask, axis=2)
        # original image
        image_np = np.array(image).astype(np.float32)
        image_np[crop_top: crop_bottom, crop_left:crop_right] *= (1 - blurred_face_mask)

        generated_images = []
        for i, regenerated_painted_face_image in enumerate(generated_image.images):
            regenerated_painted_face_image.save(f"save_dir/regenerated_painted_face_image_{i}.png")
            result_image = image_np.copy().astype(np.float32)
            
            # TODO: Fix color correction
            # If original face is brighter than generated face, the image will be too bright
            # If not use color correction, the image will be too dark if original face is brighter than generated face
            if brightnesses[i] < conf.BRIGHTNESS_THRESHOLD:
                regenerated_painted_face_image = self._apply_color_correction(regenerated_painted_face_image, cropped_face)
            
            # Smoothing to avoid sharp edges
            regenerated_painted_face_image = regenerated_painted_face_image.resize(
                (crop_right - crop_left, crop_bottom - crop_top)
            )
            regenerated_painted_face_image = np.array(regenerated_painted_face_image).astype(np.float32)
            regenerated_painted_face_image *= blurred_face_mask.astype(np.float32)

            # Fill the face area
            result_image[crop_top: crop_bottom, crop_left:crop_right] = regenerated_painted_face_image + image_np[crop_top: crop_bottom, crop_left:crop_right] 

            result_image = result_image.astype(np.uint8)
            result_image = Image.fromarray(result_image)
            generated_images.append(result_image)
        
        return generated_images
    
    def _inpainting(self, locon_inference_dto: DDSDInferenceDTO) -> List[Image.Image]:
        self.inpainting_pipe.load_lora_weights(
            locon_inference_dto.lora_path,
        )
        logger.debug("Loaded Locon Weights")

        image = Image.open(locon_inference_dto.image_path).convert("RGB")
        logger.debug(f"Image size: {image.size}")

        # Get bounding box
        crop_h_rate = conf.DDSD_CROP_HEIGHT_RATE
        crop_w_rate = conf.DDSD_CROP_WIDTH_RATE
        kernel_size = conf.COUPLE_BLUR_KERNEL_SIZE

        if locon_inference_dto.crop_h_rate:
            crop_h_rate = locon_inference_dto.crop_h_rate
        if locon_inference_dto.crop_w_rate:
            crop_w_rate = locon_inference_dto.crop_w_rate
        if locon_inference_dto.blur_kernel_size:
            kernel_size = locon_inference_dto.blur_kernel_size

        crop_left, crop_top, crop_right, crop_bottom, center_coord_rate = self._crop_face(
            image, crop_height_rate=crop_h_rate, crop_width_rate=crop_w_rate
        )

        generated_images = self._inpaint_face(
            locon_inference_dto,
            [image] * locon_inference_dto.num_outputs,
            (crop_left, crop_top, crop_right, crop_bottom),
            (kernel_size, kernel_size),
            center_coord_rate=center_coord_rate
        )
        torch.cuda.empty_cache()
        return generated_images
    
    def _increase_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        height_rate: float = 2.0,
        width_rate: float = 3.0,
        width: int = None,
        height: int = None,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
        crop_left, crop_top, crop_right, crop_bottom = bbox
        center_x = crop_left + (crop_right - crop_left) // 2
        center_y = crop_top + (crop_bottom - crop_top) // 2
        crop_h_size = int((crop_bottom - crop_top) * height_rate)
        crop_w_size = int((crop_right - crop_left) * width_rate)

        crop_left = max(0, center_x - crop_w_size // 2)
        crop_top = max(0, center_y - crop_h_size // 2)
        crop_right = min(width, crop_left + crop_w_size)
        crop_bottom = min(height, crop_top + crop_h_size)

        center_x -= crop_left
        center_y -= crop_top

        center_x_rate = center_x / (crop_right - crop_left)
        center_y_rate = center_y / (crop_bottom - crop_top)

        return (crop_left, crop_top, crop_right, crop_bottom), (center_x_rate, center_y_rate)

    def _couple(self, locon_inference_dto: CoupleInferenceDTO) -> List[Image.Image]:
        logger.debug("Loaded Locon Weights")

        image = Image.open(locon_inference_dto.image_path).convert("RGB")
        w, h = image.size
        logger.debug(f"Image size: {image.size}")
        images = [image] * locon_inference_dto.num_outputs

        crop_h_rate = conf.COUPLE_CROP_HEIGHT_RATE
        crop_w_rate = conf.COUPLE_CROP_WIDTH_RATE
        kernel_size = conf.COUPLE_BLUR_KERNEL_SIZE
        if locon_inference_dto.crop_h_rate:
            crop_h_rate = locon_inference_dto.crop_h_rate
        if locon_inference_dto.crop_w_rate:
            crop_w_rate = locon_inference_dto.crop_w_rate
        if locon_inference_dto.blur_kernel_size:
            kernel_size = locon_inference_dto.blur_kernel_size
        logger.debug(f"crop_h_rate: {crop_h_rate}, crop_w_rate: {crop_w_rate}, kernel_size: {kernel_size}")        
        for lora_path, bbox in locon_inference_dto.inpaint_info:
            logger.debug(f"bbox: {bbox}")
            # Incease bbox size
            bbox, center_coord_rate = self._increase_bbox(
                bbox,
                crop_h_rate,
                crop_w_rate,
                w,
                h,
            )
            logger.debug(f"bbox: {bbox}")
            self.inpainting_pipe.load_lora_weights(
                lora_path,
                use_safetensors=True,
            )
            logger.debug("Loaded Locon Weights")
            images = self._inpaint_face(
                locon_inference_dto,
                images,
                tuple(bbox),
                kernel_size=(kernel_size, kernel_size),
                center_coord_rate=center_coord_rate
            )
        return images

    def _i2i(self, locon_inference_dto: IMG2IMGInferenceDTO) -> List[Image.Image]:
        self.img2img_pipe.load_lora_weights(
            locon_inference_dto.lora_path,
            use_safetensors=True,
        )

        logger.debug("Loaded Locon Weights")
        image = Image.open(locon_inference_dto.image_path).convert("RGB")
    
        logger.debug(f"Image size: {image.size}")

        logger.debug("Loaded Image")
        self.img2img_pipe.scheduler = make_scheduler(
            locon_inference_dto.scheduler, self.img2img_pipe.scheduler.config
        )
        logger.debug("Loaded Scheduler")

        if locon_inference_dto.disable_safety_check:
            self.img2img_pipe.safety_checker = None

        generator = torch.Generator(conf.DEVICE)

        logger.debug(locon_inference_dto.prompt)
        logger.debug(locon_inference_dto.negative_prompt)
        prompt_embeds = self.compel(locon_inference_dto.prompt)
        negative_prompt_embeds = self.compel(locon_inference_dto.negative_prompt)

        logger.debug("Loaded Prompt Embeds")
        generated_images = []

        for i, param in enumerate(locon_inference_dto.params):
            generator.manual_seed(locon_inference_dto.seed + i)

            generated_image = self.img2img_pipe(
                prompt=[locon_inference_dto.prompt] * locon_inference_dto.num_outputs,
                negative_prompt=[locon_inference_dto.negative_prompt] * locon_inference_dto.num_outputs,
                num_inference_steps=param.num_inference_steps,
                guidance_scale=param.guidance_scale,
                generator=generator,
                strength=param.prompt_strength,
                image=image
            )
            generated_images.append(generated_image.images[0])

        torch.cuda.empty_cache()
        return generated_images
    
    def _t2i(self, locon_inference_dto: TXT2IMGInferenceDTO):
        self.txt2img_pipe.load_lora_weights(
            locon_inference_dto.lora_path,
            use_safetensors=True,
        )

        logger.debug("Loaded Locon Weights")

        logger.debug("Loaded Image")
        self.txt2img_pipe.scheduler = make_scheduler(
            locon_inference_dto.scheduler, self.img2img_pipe.scheduler.config
        )
        logger.debug("Loaded Scheduler")

        if locon_inference_dto.disable_safety_check:
            self.txt2img_pipe.safety_checker = None

        generator = torch.Generator(conf.DEVICE)

        logger.debug(locon_inference_dto.prompt)
        logger.debug(locon_inference_dto.negative_prompt)
        prompt_embeds = self.compel(locon_inference_dto.prompt)
        negative_prompt_embeds = self.compel(locon_inference_dto.negative_prompt)

        logger.debug("Loaded Prompt Embeds")
        generated_images = []

        for i, param in enumerate(locon_inference_dto.params):
            generator.manual_seed(locon_inference_dto.seed + i)

            generated_image = self.txt2img_pipe(
                prompt=[locon_inference_dto.prompt] * locon_inference_dto.num_outputs,
                negative_prompt=[locon_inference_dto.negative_prompt] * locon_inference_dto.num_outputs,
                num_inference_steps=param.num_inference_steps,
                guidance_scale=param.guidance_scale,
                generator=generator,
                width=locon_inference_dto.width,
                height=locon_inference_dto.height,
            )
            generated_images.append(generated_image.images[0])

        torch.cuda.empty_cache()
        return generated_images

    def _crop_face(self, image: Image.Image, crop_height_rate: float = 1.3, crop_width_rate: float = 1.1):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        image = ImageOps.exif_transpose(image)
        w, h = image.size

        results = face_detection.process(np.array(image))

        if results.detections and len(results.detections) == 1:
            detection = results.detections[0]
            left = int(detection.location_data.relative_bounding_box.xmin * w)
            top = int(detection.location_data.relative_bounding_box.ymin * h)
            face_w = int(detection.location_data.relative_bounding_box.width * w)
            face_h = int(detection.location_data.relative_bounding_box.height * h)

            center_x = left + face_w // 2
            center_y = top + face_h // 2
            
            crop_h_size = int(face_h * crop_height_rate)
            crop_w_size = int(face_w * crop_width_rate)

            crop_left = max(0, center_x - crop_w_size // 2)
            crop_top = max(0, center_y - crop_h_size // 2)
            crop_right = min(w, crop_left + crop_w_size)
            crop_bottom = min(h, crop_top + crop_h_size)

            crop_x = crop_right - crop_left
            crop_y = crop_bottom - crop_top

            if crop_x > crop_y:
                crop_left += (crop_x - crop_y) // 2
                crop_right = crop_left + crop_y
            elif crop_x < crop_y:
                crop_top += (crop_y - crop_x) // 2
                crop_bottom = crop_top + crop_x
            
            center_x -= crop_left
            center_y -= crop_top

            center_x_rate = center_x / (crop_right - crop_left)
            center_y_rate = center_y / (crop_bottom - crop_top)

            return crop_left, crop_top, crop_right, crop_bottom, (center_x_rate, center_y_rate)

        elif results.detections and len(results.detections) > 1:
            raise ValueError("Too many faces detected")
        else:
            raise ValueError("Face not detected")
        
    def _extract_blurred_face_mask(self, cropped_face, kernel_size, center_coord_rate=(0.5, 0.5)):
        logger.info(f"coord: {center_coord_rate}")
        mask = self.segmentator(np.asarray(cropped_face), [FACE], center_coord_rate)
        mask = mask > conf.DDSD_FACE_SEG_THRESHOLD
        brightness = np.mean(np.array(cropped_face)[mask == 1])
        logger.info(f"brightness: {brightness}")
        mask = mask.astype(np.float32)
        import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.savefig("save_dir/mask.png")

        for _ in range(3):
            mask = cv2.GaussianBlur(mask, kernel_size, 0)
        mask = cv2.dilate(mask, (5, 5), iterations=1)

        return mask, brightness
        
    def _setup_color_correction(self, image: Image.Image):
        correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
        return correction_target

    def _apply_color_correction(self, newimage: Image.Image, original_image: Image.Image) -> Image.Image:
        correction = self._setup_color_correction(original_image)
        image = Image.fromarray(
            cv2.cvtColor(
                exposure.match_histograms(
                    cv2.cvtColor(
                        np.asarray(newimage),
                        cv2.COLOR_RGB2LAB
                    ),
                    correction,
                    channel_axis=2
                ), 
                cv2.COLOR_LAB2RGB
            ).astype("uint8")
        )
        return image