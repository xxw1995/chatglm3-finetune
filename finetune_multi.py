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


tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
device_map_dict = {'transformer.word_embeddings': 0,
                   'transformer.layers.0': 0,
                   'transformer.layers.1': 0,
                   'transformer.layers.2': 0,
                   'transformer.layers.3': 0,
                   'transformer.layers.4': 0,
                   'transformer.layers.5': 0,
                   'transformer.layers.6': 0,
                   'transformer.layers.7': 0,
                   'transformer.layers.8': 0,
                   'transformer.layers.9': 0,
                   'transformer.layers.10': 0,
                   'transformer.layers.11': 0,
                   'transformer.layers.12': 0,
                   'transformer.layers.13': 0,
                   'transformer.layers.14': 0,
                   'transformer.layers.15': 1,
                   'transformer.layers.16': 1,
                   'transformer.layers.17': 1,
                   'transformer.layers.18': 1,
                   'transformer.layers.19': 1,
                   'transformer.layers.20': 1,
                   'transformer.layers.21': 1,
                   'transformer.layers.22': 1,
                   'transformer.layers.23': 1,
                   'transformer.layers.24': 1,
                   'transformer.layers.25': 1,
                   'transformer.layers.26': 1,
                   'transformer.layers.27': 1,
                   'transformer.final_layernorm': 1,
                   'lm_head': 1
                   }

class GLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(**inputs).loss
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        saved_params = {
            k: v.to("cuda") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "chatglm-lora.pt"))

def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        "model", load_in_8bit=False, trust_remote_code=True).half()

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
    model = get_peft_model(model, peft_config).to("cuda")
    for k, v in device_map_dict.items():
        if k == 'transformer.word_embeddings':
            model.transformer.embedding = model.transformer.embedding.to(f'cuda:{v}')
        if k.find("transformer.layers") != -1:
            sub_value = int(k.replace("transformer.layers.", ""))
            model.transformer.encoder.layers[sub_value] = model.transformer.encoder.layers[sub_value].to(f'cuda:{v}')
        if k == "transformer.final_layernorm":
            model.transformer.encoder.final_layernorm = model.transformer.encoder.final_layernorm.to(f'cuda:{v}')
    model.print_trainable_parameters()

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    # start train
    trainer = GLMTrainer(
        model=model,
        data_collator=data_collator,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
