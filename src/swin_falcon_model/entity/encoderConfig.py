from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union
import os
@dataclass
class EncoderConfig:
    root_dir:Path
    base_model_path:Union[str, bytes, os.PathLike]
    updated_base_model_path : Path
    params_name_or_path:str
    params_image_size: List
    params_align_long_axis:bool
    params_window_size:int
    params_encoder_layer: List
    params_learning_rate:int