from typing import Optional
from typing import List
from typing import Dict
from typing import Any

from pydantic import BaseModel


class LoconTrainRequestInput(BaseModel):
    instance_data: str
    class_data: Optional[str] = None
    seed: Optional[int] = None
    resolution: int = 512
    train_batch_size: int = 4
    max_train_steps: int = 880
    learning_rate: float = 0.00002
    lr_schedule: str = 'constant_with_warmup'
    lr_warmup_steps: int = 88
    use_8bit: bool = False
    min_snr_gamma: int = 1

class LoconTrainRequest(BaseModel):
    input: LoconTrainRequestInput
    webhook: str
    version: Optional[str] = None
    model_type: str = "bra_v6"
    request_type: Optional[str] = "TRAIN"

class TrainDTO(BaseModel):
    train_data_dir: str
    class_data_dir: Optional[str] = None
    seed: Optional[int] = None
    resolution: int = 512
    train_batch_size: int = 4
    max_train_steps: int = 880
    learning_rate: float = 0.00002
    lr_scheduler: str = 'constant_with_warmup'
    lr_warmup_steps: int = 88
    use_8bit: bool = False
    min_snr_gamma: int = 1
    outdir: str = "output"
    pretrained_model_name_or_path: str
    vae_model_path: str
    
