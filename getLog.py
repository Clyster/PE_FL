import re

def extract_summary_from_log(log_file_path):
    # 定义完整的正则表达式以匹配所需的总结性文本块
    summary_pattern = re.compile(
        r'\*\nSummary of\s+val\s+round\s*\n'
        r'RMSE=(.*?)\n'
        r'MAE=(.*?)\n'
        r'Photo=(.*?)\n'
        r'iRMSE=(.*?)\n'
        r'iMAE=(.*?)\n'
        r'squared_rel=(.*?)\n'
        r'silog=(.*?)\n'
        r'Delta1=(.*?)\n'
        r'REL=(.*?)\n'
        r'Lg10=(.*?)\n'
        r't_GPU=(.*?)\n'
        r'New best model by rmse \(was (.*?)\)\n'
        r'\*\n'
        r'Global weights: global epoch (\d+) Saved!',
        re.DOTALL
    )
    
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # 使用正则表达式查找所有匹配的总结性文本块
    summaries = summary_pattern.findall(log_content)

    return summaries

def save_summaries_to_file(summaries, output_file_path):
    with open(output_file_path, 'w') as file:
        for i, summary in enumerate(summaries, 1):
            file.write(f"Summary {i}:\n")
            file.write(f"RMSE: {summary[0]}\n")
            file.write(f"MAE: {summary[1]}\n")
            file.write(f"Photo: {summary[2]}\n")
            file.write(f"iRMSE: {summary[3]}\n")
            file.write(f"iMAE: {summary[4]}\n")
            file.write(f"squared_rel: {summary[5]}\n")
            file.write(f"silog: {summary[6]}\n")
            file.write(f"Delta1: {summary[7]}\n")
            file.write(f"REL: {summary[8]}\n")
            file.write(f"Lg10: {summary[9]}\n")
            file.write(f"t_GPU: {summary[10]}\n")
            file.write(f"Previous best RMSE: {summary[11]}\n")
            file.write(f"Global epoch: {summary[12]}\n")
            file.write("\n")

# 示例用法
log_file_path = 'path_to_your_log_file.log'
output_file_path = 'summary_output.txt'

summaries = extract_summary_from_log(log_file_path)
save_summaries_to_file(summaries, output_file_path)

print(f"Summaries have been saved to {output_file_path}")
