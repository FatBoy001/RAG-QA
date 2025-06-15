import LLMDataGenerator
import FileProcesser
import os
import time

print("Start processing report file")
path = os.path.join("test.md") 
path = os.path.abspath(path)
dir_path= os.path.dirname(path)

report = FileProcesser.load_report(path)
chunks = FileProcesser.token_aware_chunking(report,max_tokens=512,stride=10)
FileProcesser.write_to_jsonl(chunks,dir_path,"chunks")

debug_llm_replys = []
print("Generating data...")
train_data = []
for idx,chunk in enumerate(chunks):
    start = time.time()
    print(f"Processing Chunk {idx+1}/{len(chunks)}. Use time: ", end="")
    res = LLMDataGenerator.generate_data(chunk)
    debug_llm_replys.append(f"====={idx}=====")
    debug_llm_replys.append(res)
    train_data.extend(LLMDataGenerator.clean_text(res,chunk))
    print(f"{time.time() - start:.4f} sec")

FileProcesser.write_to_jsonl(debug_llm_replys,dir_path,"LLMReply")
res = LLMDataGenerator.generate_data(chunks[0])
with open("response.txt",'w') as f:
    f.write(res)
    
print("Data generated")

file_name = "train_data"
FileProcesser.write_to_jsonl(train_data,dir_path,file_name)
print(f"Data write in to jsonl name: {file_name}")
