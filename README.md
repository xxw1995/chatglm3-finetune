# chatglm3-finetune
最容易上手的0门槛chatglm3项目

## 安装依赖
pip3 install -r requirements.txt

## 处理数据
python cover_alpaca2jsonl.py  --data_path ./alpaca_data.json  --save_path ./alpaca_data.jsonl
python tokenize_dataset_rows.py  --jsonl_path ./alpaca_data.jsonl --save_path ./alpaca  --max_seq_length 200

## finetune
python finetune.py --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --max_steps 52000 --save_steps 1000 --save_total_limit 20 --learning_rate 1e-4 --remove_unused_columns false --logging_steps 50 --output_dir output 
