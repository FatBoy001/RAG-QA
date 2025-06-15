import LLMDataProcesser
import FileUtility
import os
import time
import Config

def generate(file_name,file_path):
    print(f"\n***Start processing report file {file_name}***")
    report_path = os.path.join(file_path,file_name) 
    # path = os.path.abspath(path)
    result_dir_path = os.path.join("Data")
    log_dir_path = os.path.join(Config.current_directory,"Logs")
    print(f"""Fatching data from: {os.path.abspath(report_path)}
Result will be save at: {os.path.abspath(result_dir_path)}
Log will be save at: {os.path.abspath(log_dir_path)}
    """)

    report = FileUtility.load_report(report_path)
    chunks = FileUtility.token_aware_chunking(report,max_tokens=512,stride=10)
    FileUtility.write_to_jsonl(chunks,log_dir_path,f"{file_name}_chunks")

    debug_llm_replys = []
    print("Generating data...")
    train_data = []
    for idx,chunk in enumerate(chunks):
        start = time.time()
        print(f"Processing Chunk {idx+1}/{len(chunks)}. Use time: ", end="")
        res = LLMDataProcesser.LLM_prompt_response(chunk)
        debug_llm_replys.append(f"====={idx}=====")
        debug_llm_replys.append(res)
        train_data.extend(LLMDataProcesser.generate(res,chunk))
        print(f"{time.time() - start:.4f} sec")

    FileUtility.write_to_jsonl(debug_llm_replys,log_dir_path,f"{file_name}_LLMReply")
    res = LLMDataProcesser.LLM_prompt_response(chunks[0])
    with open("response.txt",'w') as f:
        f.write(res)

    print("Data generated")

    file_name = f"{file_name}_data"
    FileUtility.write_to_jsonl(train_data,result_dir_path,file_name)
    print(f"Result data write in to jsonl name: {file_name}")


# 執行生成
reports_path = os.path.join(Config.current_directory,"Report","Markdown")
reports_name = os.listdir(reports_path)

for report_name in reports_name:
    generate(report_name,reports_path)