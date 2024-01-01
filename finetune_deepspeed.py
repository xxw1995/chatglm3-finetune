from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from model.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import datasets
import os
from argument import FinetuneArguments, CastOutputToFloat
import deepspeed

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class GLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(**inputs).loss
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        saved_params = {
            k: v for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "chatglm-lora.pt"))

def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        #"model", load_in_8bit=False, trust_remote_code=True, device_map="auto"
        "model", load_in_8bit=False, trust_remote_code=True
    ).half()

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.transformer.output_layer)
    model.config.use_cache = (
        False
    )
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        target_modules=['query_key_value'],
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    print("**未启用deepspeed**")
    model.print_trainable_parameters()
    #"""
    # stage1
    conf = {"train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": 50
            }
    #"""

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                    model=model,
                                    model_parameters=model.parameters())
    model_engine.train()
    print("**启用deepspeed**")

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    # start train
    trainer = GLMTrainer(
        model=model_engine,
        data_collator=data_collator,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
