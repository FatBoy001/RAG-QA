# from langchain.text_splitter import CharacterTextSplitter
import json
import os
from transformers import AutoTokenizer
import torch

def load_report(pdf_report_path:str)->str:
    with open(pdf_report_path, 'r') as f:
        financial_text = f.read()
    return financial_text

# def split_to_chunks(texts,chunk_size=2000,chunk_overlap=100)->list[str]:
#     splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
#     res = splitter.split_text(texts)
#     return res

def token_aware_chunking(text, max_tokens=512, stride=100):
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
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

def write_to_md(text,dir_pathpath:str,file_name="test"):
    write_path = os.path.join(dir_pathpath,f"{file_name}.md")
    with open(write_path, 'w') as f:
        f.write(text)

def write_to_txt(text,dir_pathpath:str,file_name="test"):
    write_path = os.path.join(dir_pathpath,f"{file_name}.txt")
    with open(write_path, 'w') as f:
        f.write(text)

# pdf_text = extract_text_from_pdf("/home/kuo/ReportAnalysis/NVIDIAAn.pdf")
# # # md_text = pymupdf4llm.to_markdown("NVIDIAAn.pdf")
# write_to_txt(pdf_text,"/home/kuo/ReportAnalysis/")
