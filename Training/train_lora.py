# 讀取資料集
from datasets import load_dataset
import Config

data_path = "/home/kuo/RAG-QA/DatasetGenerator/Data/train_data.jsonl"
data_files = {
    "train": [
        "/home/kuo/RAG-QA/Data/2024_Q2_NVIDIA.pdf.md_data.jsonl",
        "/home/kuo/RAG-QA/Data/2024_Q3_NVIDIA.pdf.md_data.jsonl"
    ],

    "validation": "/home/kuo/RAG-QA/Data/2024_Q4_NVIDIA.pdf.md_data.jsonl",
    "test": "/home/kuo/RAG-QA/Data/2023_Q1_NVIDIA.pdf.md_data.jsonl"
}
print(f"Loading data from {data_files}")

datasets = load_dataset(
    "json",
    data_files=data_files,
)
train_dataset = datasets["train"]
val_dataset = datasets["validation"]
test_dataset = datasets["test"]
# 資料整理
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
tokenizer.pad_token = tokenizer.eos_token
max_token_length = 700
print(f"Tokenizing data max token length: {max_token_length}")
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
    # 需要注意長度否則資訊會不完整
    tokenized = tokenizer(prompt,max_length=max_token_length,truncation=True,padding="max_length",return_tensors=None )
    # Trainer會調用Labels
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train_dataset = train_dataset.map(tokenize)
tokenized_val_dataset = val_dataset.map(tokenize)

# 讀取model
from transformers import AutoModelForCausalLM
from peft import get_peft_model
import torch
print(f"""
Traning info:
    Model name: {Config.model_name}
    LoRA Config:
        r = {Config.lora_config.r}
        target_modules = {Config.lora_config.target_modules}
        task_type= {Config.lora_config.task_type}
    Traning Setting:
        epochs = {Config.training_args_config.num_train_epochs}
        learning_rate = {Config.training_args_config.learning_rate}
        per_device_train_batch_size (batch_size) = {Config.training_args_config.per_device_train_batch_size}
        gradient_accumulation_steps = {Config.training_args_config.gradient_accumulation_steps}
""")
model = AutoModelForCausalLM.from_pretrained(
    Config.model_name,
    load_in_8bit=True,      # 或改成 4-bit（load_in_4bit=True）
    device_map="auto",
    torch_dtype=torch.float16
)


# 導入設定
# LoRA設定
model = get_peft_model(model, Config.lora_config)
# train
from transformers import Trainer
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=Config.training_args_config,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
print("*** Start Training ***")
trainer.train()