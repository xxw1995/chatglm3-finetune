# chatglm3-finetune
0门槛的 **chatglm3-finetune & agent** 项目  
已经支持 **基于langchain的agent调用** & **根据zero_shot的LLM** & **知识库召回知识** 完成 **intent识别**  

**注意**  
  1.model_32K版本需要特殊的数据格式和loss_mask，本项目暂时支持model_base版本，使用download_model.py下载即可  
  2.非base版本的agent需要special_token，本项目暂时支持model_base版本的agent调用  


## 更新
TODO:增加rank功能，去噪音。  
2024.1.1更新：支持deepspeed多卡训练：python -m torch.distributed.launch --nproc_per_node=2  finetune_deepspeed.py   --dataset_path ./alpaca --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 52000 --save_steps 1000 --save_total_limit 20 --learning_rate 1e-4 --remove_unused_columns false --logging_steps 50 --output_dir output  
2023.12.16更新：增加了一种新的调用下游tools的方式（function_map）;在输入文本过长的时候使用LLM做向量召回。当知识库没有召回任何东西的时候，使用对齐后的chat模型生成知识（后续替换为gpt4），再做回答（可以利用下游的tools）。  
2023.12.5更新：支持bge进行知识库的知识召回，在agent模块中的意图识别bge使用single模式（不支持多意图）、辅助LLM做生成的bge使用multi模式（多条知识召回）  
2023.11.16更新：已支持多卡 finetune_multi.py  


## 安装依赖
pip3 install -r requirements.txt  

## 下载模型
python download_model.py 

## 处理数据
python cover_alpaca2jsonl.py  --data_path ./alpaca_data.json  --save_path ./alpaca_data.jsonl  
python tokenize_dataset_rows.py  --jsonl_path ./alpaca_data.jsonl --save_path ./alpaca  --max_seq_length 200  

## Finetune
python finetune.py --dataset_path ./alpaca --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 52000 --save_steps 1000 --save_total_limit 20 --learning_rate 1e-4 --remove_unused_columns false --logging_steps 50 --output_dir output  

## Inference
python infer.py output/checkpoint-1000 ./alpaca_data.json 100 

## Agent  
python run.py  


## 感谢
https://github.com/mymusise/ChatGLM-Tuning  
https://github.com/THUDM/ChatGLM3  
https://github.com/minghaochen/chatglm3-base-tuning

