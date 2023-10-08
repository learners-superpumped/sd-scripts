from typing import Optional
from typing import List
from typing import Dict
from typing import Any
from typing import Tuple

from pydantic import BaseModel

class Params(BaseModel):
    guidance_scale: float
    num_inference_steps: int
    prompt_strength: Optional[float] = None


class LoconInferenceRequestInput(BaseModel):
    disable_safety_check: bool
    image: Optional[str] = None
    locon_url: Optional[str] = None
    scheduler: str
    weights: Optional[str] = None
    negative_prompt: str
    num_outputs: Optional[int]
    prompt: str
    params: List[Params]
    seed: Optional[int] = None
    width: int
    height: int
    type: Optional[str] = "image2image" # image2image, ddsd
    ddsd: Optional[bool] = False # will be removed
    repaint: Optional[bool] = True # only for ddsd
    inpaint_info: List[List[Any]] = None # only
    crop_w_rate: Optional[float] = None # only for ddsd, couple
    crop_h_rate: Optional[float] = None # only for ddsd, couple
    blur_kernel_size: Optional[int] = None # only for ddsd, couple

class SDXLInferenceRequestInput(BaseModel):
    lora_url: str
    embedding_url: str
    prompt: str
    negative_prompt: Optional[str] = None
    num_outputs: Optional[int]
    scheduler: str
    width: int
    height: int
    image: str
    disable_safety_check: Optional[bool] = None # TODO: remove this
    params: Optional[List[Params]] = [] # TODO: remove this
    seed: Optional[int] = None
    type: Optional[str] = "image2image" # only support image2image 

# TODO: Remove this
class ComfyInferenceRequestInput(BaseModel):
    lora_url: str
    prompt: Dict[str, Any]
    num_outputs: Optional[int]
    image: str

class LoconInferenceRequest(BaseModel):
    version: Optional[str]
    input: LoconInferenceRequestInput
    webhook: str

class SDXLInferenceRequest(BaseModel):
    version: Optional[str]
    input: SDXLInferenceRequestInput
    webhook: str

# TODO: Remove this
class ComfyInferenceRequest(BaseModel):
    version: Optional[str]
    input: ComfyInferenceRequestInput
    webhook: str

class IMG2IMGInferenceDTO(BaseModel):
    image_path: str
    lora_path: str
    embedding_path: Optional[str] = None
    disable_safety_check: Optional[bool] = False
    params: List[Params]
    scheduler: str
    negative_prompt: Optional[str] = None
    num_outputs: Optional[int]
    prompt: str
    height: int
    width: int
    seed: int

class TXT2IMGInferenceDTO(BaseModel):
    lora_path: str
    embedding_path: Optional[str] = None
    disable_safety_check: Optional[bool] = False
    params: List[Params]
    scheduler: str
    negative_prompt: Optional[str] = None
    num_outputs: Optional[int]
    prompt: str
    height: int
    width: int
    seed: int

class DDSDInferenceDTO(IMG2IMGInferenceDTO):
    repaint: Optional[bool] = False
    crop_w_rate: Optional[float] = None # only for ddsd, couple
    crop_h_rate: Optional[float] = None # only for ddsd, couple
    blur_kernel_size: Optional[int] = None # only for ddsd, couple

class CoupleInferenceDTO(IMG2IMGInferenceDTO):
    repaint: Optional[bool] = False
    inpaint_info: List[Tuple[str, Tuple[int, int, int, int]]] = None # only
    crop_w_rate: Optional[float] = None # only for ddsd, couple
    crop_h_rate: Optional[float] = None # only for ddsd, couple
    blur_kernel_size: Optional[int] = None # only for ddsd, couple

# TODO: Remove this
class ComfyInferenceDTO(BaseModel):
    image_path: str
    lora_path: str
    prompt: Dict[str, Any]
    num_outputs: Optional[int]
