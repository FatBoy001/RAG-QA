model_name="meta-llama/Llama-3.2-3B-Instruct"

from peft import LoraConfig
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]  # 適用 LLaMA 3 架構
)

from transformers import TrainingArguments
training_args_config = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    per_device_train_batch_size=4, # 實際放入的batch大小
    # gradient_accumulation_steps=2, # 每累積幾次之後才更新
    num_train_epochs=30,
    learning_rate=0.0001,
    fp16=True,
    output_dir="./Models",
    logging_dir="./Logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=100
)