from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA 要設定 pad token

adapter_path = "Models/checkpoint-119"  # 你剛剛產生的資料夾
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

from datasets import load_dataset
data_path = "/home/kuo/RAG-QA/Data/2023_Q1_NVIDIA.pdf.md_data.jsonl"
print(f"Loading data from {data_path}")
dataset = load_dataset(
    "json",
    data_files=data_path,
    split="train"
)

# 你可以使用 tokenizer 的 chat 模板
content = f"Concise reply the following question:\n {dataset[0]['instruction']}\n\n{dataset[0]['input']}"
print(content)
messages = [
    {"role": "user", "content":content}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成回答
print("\n\n\n*** Answer ***")
output = model.generate(**inputs, max_new_tokens=200)
input_ids = inputs["input_ids"]
generated_ids = output[0][input_ids.shape[1]:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(answer)

