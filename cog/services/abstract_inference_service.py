from abc import ABC, abstractmethod
from typing import Optional, List
from PIL import Image

class InferenceService(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(
        self,
        prompt: str,
        image_path: str,
        lora_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_outputs: int = 1,
        scheduler: str = 'euler_a',
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        prompt_strength: float = 0.55,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        pass