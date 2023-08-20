from dataclasses import dataclass
@dataclass
class AdapterModuleConfig:
    params_in_features: int
    params_hidden_dim :int
    params_groups: int
    params_scale: int