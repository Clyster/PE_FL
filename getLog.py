import re

def extract_summary_from_log(log_file_path):
    summaries = []

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 搜索 "Summary of val round" 并捕获接下来的 15 行
    i = 0
    while i < len(lines):
        if "Summary of  val round" in lines[i]:
            # 捕获当前行和接下来的 n 行
            summary = ''.join(lines[i:i+17])
            summaries.append(summary)
            i += 17  # 跳过已经捕获的行
        else:
            i += 1

    return summaries

def save_summaries_to_file(summaries, output_file_path):
    with open(output_file_path, 'w') as file:
        for summary in summaries:
            file.write(summary)
            file.write('\n\n')


# 示例用法
log_file_path = '/home/ubuntu-user/Desktop/hzy/1.txt'
output_file_path = '/home/ubuntu-user/Desktop/hzy/summary_output.txt'

summaries = extract_summary_from_log(log_file_path)
save_summaries_to_file(summaries, output_file_path)

print(f"Summaries have been saved to {output_file_path}")
