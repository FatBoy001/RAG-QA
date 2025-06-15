import json
import os
from transformers import AutoTokenizer
import torch
import Config
def load_report(report_path:str)->str:
    """
    讀取文字檔案
    Args:
        report_path (str): 檔案路徑
    Returns:
        str: 文件中的文字
    """
    
    with open(report_path, 'r') as f:
        financial_text = f.read()
    return financial_text

def token_aware_chunking(text, max_tokens=512, stride=100):
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        # cache_dir="/mnt/nvme0/kuo/hg_models",
        torch_dtype=torch.float16
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - stride  # overlapping by stride tokens

    return chunks

def write_to_jsonl(list_data:list,dir_pathpath:str,file_name="data"):
    write_path = os.path.join(dir_pathpath,f"{file_name}.jsonl")
    os.makedirs(dir_pathpath, exist_ok=True)
    with open(write_path,'w') as f:
        for data in list_data:
            json.dump(data,f)
            f.write("\n")

# pdf to text

import pymupdf
import pymupdf4llm

def extract_text_from_pdf(pdf_path:str)->str:
    with pymupdf.open(pdf_path) as doc:  # open document
        text = chr(12).join([page.get_text() for page in doc])
    return text

def extract_md_from_pdf(pdf_path:str)->str:
    md_text=pymupdf4llm.to_markdown(pdf_path)
    return md_text

def write_to_md(text,dir_path:str,file_name="test"):
    write_path = os.path.join(dir_path,f"{file_name}.md")
    with open(write_path, 'w') as f:
        f.write(text)

def write_to_txt(text,dir_path:str,file_name="test"):
    write_path = os.path.join(dir_path,f"{file_name}.txt")
    with open(write_path, 'w') as f:
        f.write(text)

# pdf_text = extract_text_from_pdf("/home/kuo/ReportAnalysis/NVIDIAAn.pdf")
# # # md_text = pymupdf4llm.to_markdown("NVIDIAAn.pdf")
# write_to_txt(pdf_text,"/home/kuo/ReportAnalysis/")
