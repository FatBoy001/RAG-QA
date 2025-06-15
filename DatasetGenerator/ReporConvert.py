import FileUtility
import Config
import os

reports_path = os.path.join(Config.current_directory,"Report","Pdf")
result_path = os.path.join(Config.current_directory,"Report","Markdown")
files_name = os.listdir(reports_path)
print("Directory info:")
for f in files_name:
    print(f)
print("\nConverting to .md")
for report_name in files_name:
    report_pdf_path = os.path.join(reports_path,report_name)
    print(f"Processing {report_pdf_path}")
    pdf_text=FileUtility.extract_md_from_pdf(report_pdf_path)
    FileUtility.write_to_md(pdf_text,result_path ,file_name=report_name)