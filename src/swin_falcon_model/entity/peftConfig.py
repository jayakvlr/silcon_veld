import torch

class PeftConfig:
    def __init__(self,load_in_4bit:bool=True,
                 bnb_4bit_use_double_quant:bool=True,
                 bnb_4bit_quant_type :str="nf4",
                 bnb_4bit_compute_dtype:torch.dtype =torch.float16):
        self.load_in_4bit=load_in_4bit
        self.bnb_4bit_use_double_quant=bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type=bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype=bnb_4bit_compute_dtype # add a colon here
