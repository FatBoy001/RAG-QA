from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import gc

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct"#,
    #cache_dir="/mnt/nvme0/kuo/hg_models",
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct"#,
    # cache_dir="/mnt/nvme0/kuo/hg_models"
).to("cuda")
"""
**Question:** What is the fiscal year-end date for NVIDIA Corporation?
**Answer:** The fiscal year end date for NVIDIA Corporation is <date>.

**Question:** What is the non-GAAP earnings per diluted share for NVIDIA's <year> fourth quarter?
**Answer:** The non-GAAP earnings per diluted share for NVIDIA's fourth quarter is <amount>.
"""
def generate_data(chunk:str,num_questions=5)->str:
    
    instruction = f"""You are an earnings report analyst. Your task is to make {num_questions} pair of questions and answer to understand a company, its financial report, and its key financial performance. Restrict the generated content must be based on the report.
With each pair separated by a newline, must use the following format exactly, here is the format example, your answer must not be identical to the example, but you may express it in a different way. Do not include any further explanation or follow-up:
**Question:** <your question>
**Answer:** <your answer>

Here is part of the report:
{chunk}
"""
    messages = [
        {"role": "earnings report analyst", "content": instruction}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.7,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    gc.collect()
    return output_text

def clean_text(input,chunk)->list:
    texts=input.split('\n')
    res=[]
    data={}
    for text in texts:
        if not("**Question:**" in text or "**Answer:**" in text):continue
        if "<your question>" in text or "<your answer>" in text:continue
        key = "Question" if len(data)==0 else "Answer"
        data[key] = re.sub(r"\*\*(Question|Answer):\*\*\s*","",text)

        if(len(data)!=2): continue
        data["Chunk"] = chunk
        res.append(data)
        data={}
        
    return res