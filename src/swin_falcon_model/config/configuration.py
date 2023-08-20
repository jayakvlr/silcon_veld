from pathlib import Path
from swin_falcon_model.constants import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from swin_falcon_model.entity.encoderConfig import EncoderConfig
from swin_falcon_model.entity.silconConfig import SilconConfig
from swin_falcon_model.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_prepare_base_model_config(self)->EncoderConfig:
        config=self.config.prepare_base_model
        create_directories([config.root_dir])
        base_model_config=EncoderConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_name_or_path=Path(self.params.params_name_or_path),
            params_image_size=self.params.params_image_size,
            params_window_size=self.params.params_window_size,
            params_learning_rate=self.params.params_learning_rate,
            params_encoder_layer=self.params.params_encoder_layer,
            params_align_long_axis=self.params.params_align_long_axis
        )
        return base_model_config
    def get_prepare_adapter_model_config(self)->SilconConfig:
        adapter_model_config=AdapterModuleConfig(
                params_in_features=self.params.params_in_features,
                params_hidden_dim = self.params.params_hidden_dim,
                params_groups= self.params.params_groups,
                params_scale=self.params.params_scale 
                )
        return adapter_model_config
    
    def get_prepare_full_model(self)->SilconConfig:
        full_model=SilconConfig(
            add_prefix_space=False,
            eos_token= "<|endoftext|>",
            model_max_length= 2048,
            name_or_path="tiiuae/falcon_tokenizer",
            special_tokens_map_file=None,
            tokenizer_class="PreTrainedTokenizerFast"
        )
        return full_model