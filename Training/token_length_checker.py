import Config
from transformers import AutoTokenizer
from datasets import load_dataset


data_files = {
    "train": [
        "/home/kuo/RAG-QA/Data/2024_Q2_NVIDIA.pdf.md_data.jsonl",
        "/home/kuo/RAG-QA/Data/2024_Q3_NVIDIA.pdf.md_data.jsonl"
    ],

    "validation": "/home/kuo/RAG-QA/Data/2024_Q4_NVIDIA.pdf.md_data.jsonl",
    "test": "/home/kuo/RAG-QA/Data/2023_Q1_NVIDIA.pdf.md_data.jsonl"
}
datasets = load_dataset("json", data_files=data_files)

train_dataset = datasets["train"]
val_dataset = datasets["validation"]
test_dataset = datasets["test"]

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
tokenizer.pad_token = tokenizer.eos_token
def tokenize(example):
    messages = [
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example['output']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized = tokenizer(prompt)
    # Trainer會調用Labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

import numpy as np

def cal(dataset):
    tokenized_dataset = dataset.map(tokenize)
    lengths = []
    for sample in tokenized_dataset:
        lengths.append(len(sample["input_ids"]))

    print("Average token length：", np.mean(lengths))
    print("Max token length：", np.max(lengths))
    print("90 percentage：", np.percentile(lengths, 90))

print("train_dataset")
cal(train_dataset)
print("val_dataset")
cal(val_dataset)
print("test_dataset")
cal(test_dataset)