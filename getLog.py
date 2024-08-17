import re

def extract_summary_from_log(log_file_path):
    summaries = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 搜索 "Summary of val round" 并捕获接下来的 15 行
    i = 0
    while i < len(lines):
        if "Summary of  val round" in lines[i]:
            # 捕获当前行和接下来的 15 行
            summary = ''.join(lines[i:i+16])
            summaries.append(summary)
            i += 16  # 跳过已经捕获的行
        else:
            i += 1

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
