import os


class Settings:
    PROJECT_NAME: str = "inference server"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"

    # Service
    SERVICE_NAME: str = os.environ.get("SERVICE_NAME", "LoconInferenceService")

    # INFERENCE REDIS
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: str = os.environ.get("REDIS_PORT", "6379")
    REDIS_PASSWORD: str = os.environ.get("REDIS_PASSWORD", "")
    REDIS_USERNAME: str = os.environ.get("REDIS_USERNAME", "default")
    QUEUE_LIMIT: int = int(os.environ.get("QUEUE_LIMIT", "30000"))
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 100
    REDIS_QUEUE_NAME: str = os.environ.get("QUEUE", "comfy-infer-queue")

    # BUCKET SETTINGS
    GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "learneroid")
    UPLOAD_BUCKET: str = os.environ.get("BUCKET", "stable-diffusion-server-dev")

    BASE_MODEL_BUCKET: str = os.environ.get("BASE_MODEL_BUCKET", "stable-diffusion-model-repository-asia")
    BASE_MODEL_VERSION: str = os.environ.get("BASE_MODEL_VERSION", "v1")
    BASE_MODEL_STAGE: str = os.environ.get("BASE_MODEL_STAGE", "production")
    BASE_MODEL_NAME: str = os.environ.get("BASE_MODEL_NAME", "rsp.safetensors") # before : sd_xl_base_1.0.safetensors
    BASE_MODEL_PATH: str = os.environ.get("BASE_MODEL_PATH", "./models")
    MODEL_CACHE: str = "diffusers-cache"
    DEVICE: str = os.environ.get("DEVICE", "cuda")

    # Asynchronous processing
    MAX_QUEUE_SIZE: int = int(os.environ.get("MAX_QUEUE_SIZE", "2"))

    # Cache
    MIN_DISK_FREE: int = int(os.environ.get("MIN_DISK_FREE", "1000000000")) # 1GB

    # SAM
    SAM_FOREGROUND_OFFSET: int = int(os.environ.get("SAM_FOREGROUND_OFFSET", "5"))

    # DDSD settings
    DDSD_BLUR_KERNEL_SIZE: int = int(os.environ.get("DDSD_BLUR_KERNEL_SIZE", "160"))
    DDSD_CROP_HEIGHT_RATE: float = float(os.environ.get("DDSD_CROP_HEIGHT_RATE", "2"))  
    DDSD_CROP_WIDTH_RATE: float = float(os.environ.get("DDSD_CROP_WIDTH_RATE", "2"))  
    DDSD_FACE_SEG_THRESHOLD: float = float(os.environ.get("FACE_SEG_THRESHOLD", "0.1"))
    DDSD_FACE_SEG_SMOOTH: float = float(os.environ.get("FACE_SEG_SMOOTHING", "1.0"))

    # Couple
    COUPLE_BLUR_KERNEL_SIZE: int = int(os.environ.get("COUPLE_BLUR_KERNEL_SIZE", "43"))
    COUPLE_CROP_HEIGHT_RATE: float = float(os.environ.get("COUPLE_CROP_HEIGHT_RATE", "1.5"))  
    COUPLE_CROP_WIDTH_RATE: float = float(os.environ.get("COUPLE_CROP_WIDTH_RATE", "1.5"))
    COUPLE_FACE_SEG_THRESHOLD: float = float(os.environ.get("COUPLE_FACE_SEG_THRESHOLD", "0.3"))

    # Repaint
    REPAINT: bool = os.environ.get("REPAINT", "False").lower() == "true"
    REPAINT_NUM_INFERENCE_STEPS: int = int(os.environ.get("REPAINT_NUM_INFERENCE_STEPS", "15"))
    REPAINT_PROMPT_STRENGTH: float = float(os.environ.get("REPAINT_PROMPT_STRENGTH", "0.1"))
    REPAINT_GUIDANCE_SCALE: float = float(os.environ.get("REPAINT_GUIDANCE_SCALE", "6"))

    # COLOR
    BRIGHTNESS_THRESHOLD: float = float(os.environ.get("BRIGHTNESS_THRESHOLD", "120"))
    DILATE_KERNEL_SIZE: int = int(os.environ.get("DILATE_KERNEL_SIZE", "5"))

    # Train
    TRAIN_DATA_DIR: str = os.environ.get("TRAIN_DATA_DIR", "/home/learners/roundtable/stable-diffusion-server/producer-consumer/locon_inference_consumer/data")
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/home/learners/roundtable/stable-diffusion-server/producer-consumer/locon_inference_consumer/output")
    VAE_MODEL_NAME: str = os.environ.get("VAE_MODEL_PATH", "vae-ft-mse-840000-ema-pruned.safetensors")

conf = Settings()
