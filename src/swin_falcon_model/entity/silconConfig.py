import os
from typing import List, Union
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

class SilconConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DonutModel`]. It is used to
    instantiate a Donut model according to the specified arguments, defining the model architecture

    Args:
        input_size:
            Input image size (canvas size) of Donut.encoder, SwinTransformer in this codebase
        align_long_axis:
            Whether to rotate image if height is greater than width
        window_size:
            Window size of Donut.encoder, SwinTransformer in this codebase
        encoder_layer:
            Depth of each Donut.encoder Encoder layer, SwinTransformer in this codebase
        decoder_layer:
            Number of hidden layers in the Donut.decoder, such as BART
        max_position_embeddings
            Trained max position embeddings in the Donut decoder,
            if not specified, it will have same value with max_length
        max_length:
            Max position embeddings(=maximum sequence length) you want to train
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local
    """

    model_type = "Silcon"
    def  __init__(
            self,
            input_size:List[int]=[2560,1920],
            align_long_axis: bool =False,
            window_size :int=10,
            encoder_layer:List[int] =[2,2,14,2],
            decoder_layer:int=4,
            max_position_embedding:int=None,
            max_length:int=1536,
            name_or_path :Union[str,bytes,os.PathLike]="",
            **kwargs
    ):
        super().__init__()
        self.input_size=input_size
        self.along_long_axis=align_long_axis
        self.window_size=window_size
        self.encoder_layer=encoder_layer
        self.decoder_layer=decoder_layer
        self.max_position_embedding=max_length if max_position_embedding is None else max_position_embedding
        self.max_length=max_length
        self.name_or_path=name_or_path