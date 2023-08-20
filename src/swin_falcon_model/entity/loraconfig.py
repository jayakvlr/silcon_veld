class LoraConfigArgs:

    def  __init__(self,lora_rank,
        lora_alpha:int=8,
        target_modules:int=32,
        lora_dropout:float=0.05,
        lora_bias:str=None,
        lora_task_type:str="CAUSAL_LM",
        lora_target_modules:str=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    ):
        
        self.lora_rank=lora_rank
        self.lora_alpha=lora_alpha
        self.target_modules=target_modules
        self.lora_dropout=lora_dropout
        self.lora_bias=lora_bias
        self.lora_task_type=lora_task_type
        self.lora_target_modules=lora_target_modules