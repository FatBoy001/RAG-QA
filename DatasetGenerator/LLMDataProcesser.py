from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import gc
import Config
tokenizer = AutoTokenizer.from_pretrained(
    Config.model_name
)
model = AutoModelForCausalLM.from_pretrained(
    Config.model_name
).to("cuda")


def LLM_prompt_response(chunk:str,num_questions=5)->str:
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
    # add_generation_prompt 表示需要讓model response提供解答，會在最後自動加上 <|assistant|> 或對應標記，提示模型從這裡開始生成
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 將純文字 prompt 編碼成 input_ids 和 attention_mask，return_tensors="pt" 產生 PyTorch tensors
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

def generate(LLM_response:str,chunk:str,is_sft_format=True)->list:
    """
    整理LLM的回覆，取出Question與Answer並整理成訓練用資料集
    Args:
        LLM_response (str): LLM所產生的文章
        chunk (str): 對應對LLM提供的Context Prompt包含的文字段落
        is_sft_format (str, optional): 是否將資料整理成sft格式 (default 採用sft格式)
    Returns:
        list: 陣列中包含dictionary對應QA與Chunk
    """
    texts=LLM_response.split('\n')
    res=[]
    data={}
    for text in texts:
        if not("**Question:**" in text or "**Answer:**" in text):continue
        if "<your question>" in text or "<your answer>" in text:continue
        key = "Question" if len(data)==0 else "Answer"
        data[key] = re.sub(r"\*\*(Question|Answer):\*\*\s*","",text)

        if(len(data)!=2): continue
        data["Chunk"] = chunk

        if is_sft_format: data = convert_to_sft_format(data)
        res.append(data)
        data={}
        
    return res

def convert_to_sft_format(data)->dict:
    return {
        "instruction": data["Question"],
        "input": data["Chunk"],
        "output": data["Answer"]
    }
