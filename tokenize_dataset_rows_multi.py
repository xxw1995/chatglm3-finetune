import argparse
import json
from tqdm import tqdm
import datasets
import transformers
from concurrent.futures import ProcessPoolExecutor
import concurrent
from functools import partial

def init_process(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained("./model", trust_remote_code=True, device_map='auto')
    return tokenizer, config

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids}

def process_line(model_name, max_seq_length, skip_overlength, line):
    tokenizer, config = init_process(model_name)
    example = json.loads(line)
    feature = preprocess(tokenizer, config, example, max_seq_length)
    if skip_overlength and len(feature["input_ids"]) > max_seq_length:
        return None
    feature["input_ids"] = feature["input_ids"][:max_seq_length]
    return feature

def read_jsonl_parallel(path, max_seq_length, skip_overlength, num_proc):
    features = []
    with open(path, "r") as f:
        lines = f.readlines()
    model_name = "./model"
    process_line_with_args = partial(process_line, model_name, max_seq_length, skip_overlength)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(process_line_with_args, line) for line in lines]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines)):
            result = future.result()
            if result is not None:
                features.append(result)
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/alpaca_data.jsonl")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()
    features = read_jsonl_parallel(args.jsonl_path, args.max_seq_length, args.skip_overlength, num_proc=2)


    dataset = datasets.Dataset.from_dict({'features': features})
    dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    main()
