import torch.nn as nn
import os
from typing import Union
from swin_falcon_model.entity.loraconfig import LoraConfigArgs
from swin_falcon_model.entity.peftConfig import PeftConfig
from transformers import AutoTokenizer,FalconModel, FalconConfig,BitsAndBytesConfig,Trainer, TrainingArguments, DataCollatorForLanguageModeling,EvalPrediction
from peft import prepare_model_for_kbit_training,LoraConfig ,get_peft_model
from argparse import Namespace
from datasets import DatasetDict, Dataset
from typing import Callable, Dict
from sagemaker.experiments import Run
import evaluate
from transformers.debug_utils import DebugUnderflowOverflow

class FalconDecoder(nn.Module):

    def __init__(self,
                 device_map:str="auto",
                 trust_remote_code:bool=True,
                 name_or_path:Union[str,bytes,os.PathLike]=None):
        super().__init__()

        self.tokenizer=AutoTokenizer('tiiuae/falcon-7b' if not name_or_path else name_or_path)
        self.config=FalconConfig('tiiuae/falcon-7b' if not name_or_path else name_or_path)

        bnb = PeftConfig()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bnb.load_in_4bit,
            bnb_4bit_use_double_quant=bnb.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb.bnb_4bit_compute_dtype,
        )
        self.model=FalconModel.from_pretrained(
            pretrained_model_name_or_path='tiiuae/falcon-7b' if not name_or_path else name_or_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
            device_map=self.device_map
        )
        lrca=LoraConfigArgs()
        target_modules=' '.join(lrca.target_modules)
        lora_config= LoraConfig(
        r=lrca.lora_rank,
        lora_alpha=lrca.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lrca.lora_dropout,
        bias=lrca.lora_bias,
        task_type=lrca.lora_task_type
    )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.model = get_peft_model(self.model, lora_config)

    def forward(x):
            

        # Falcon requires you to allow remote code execution. This is because the model 
        # uses a new architecture that is not part of transformers yet.
        # The code is provided by the model authors in the repo.


        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=args.mlm
        )

        target_modules = vars(args)["lora_target_modules"]
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=args.lora_task_type
        )

        model = get_peft_model(model, config)

        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=args.bf16,
            save_strategy=args.save_strategy,
            evaluation_strategy=args.eval_strategy,
            output_dir=args.output_dir,
            report_to=args.report_to,
        )


    def train(args: Namespace) -> None:
        dataset_dict = load_data(args)


        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            args=training_args,
            data_collator=data_collator,
        
        )

        model.config.use_cache = args.use_cache  # silence the warnings. Please re-enable for inference!
        
        #trainer.add_callback(SagemakerExperimentsCallback(run=run))

        # Start training
        trainer.train()

        trainer.save_model(output_dir=args.output_dir)

        trainer.evaluate()


def load_data(args: Namespace) -> DatasetDict:
    train_data = Dataset.load_from_disk(args.sm_channel_train)
    test_data = Dataset.load_from_disk(args.sm_channel_test)
    return DatasetDict({"train": train_data, "test": test_data})
