#from swin_falcon_model.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
#from swin_falcon_model.utils.common import create_directories, read_yaml
import torch.nn as nn
import torch
from torchvision.transforms.functional import resize, rotate
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from swin_falcon_model.entity.encoderConfig import EncoderConfig
from timm.models.swin_transformer import SwinTransformer
import PIL
import numpy as np
from PIL import ImageOps


class SwinEncoder(nn.Module):
    
    def __init__(self,config:EncoderConfig):


        super().__init__()
        self.config=config
        self.input_size = self.config.params_image_size
        self.align_long_axis = self.config.params_align_long_axis
        self.window_size = self.config.params_window_size
        self.encoder_layer = self.config.params_encoder_layer

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        return x
    
    def save_model(self):
        torch.save(self.state_dict,self.config.base_model_path)

    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))
    