import os
import re
import csv

# 定义源目录和目标目录
source_dir = '../GCD_VMs'
target_dir = '../GCD_VMs_new'

# 如果目标目录不存在，则创建它
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取源目录下的所有文件
file_list = os.listdir(source_dir)

# 用于存储每个 vm id 对应的文件列表
vm_files_dict = {}

# 遍历文件列表，根据 vm id 分组
for file_name in file_list:
    # 使用正则表达式匹配文件名中的 vm id 和序号
    match = re.match(r'vm_(\w+)_(\d+)', file_name)
    if match:
        vm_id = match.group(1)
        file_index = int(match.group(2))
        if vm_id not in vm_files_dict:
            vm_files_dict[vm_id] = []
        vm_files_dict[vm_id].append((file_index, file_name))

# 遍历每个 vm id 的文件列表，按序号排序并合并文件
for vm_id, files in vm_files_dict.items():
    # 按序号排序
    files.sort(key=lambda x: x[0])

    # 定义合并后的文件名
    merged_file_name = f'vm_{vm_id}_merged.csv'
    merged_file_path = os.path.join(target_dir, merged_file_name)

    # 打开合并后的 CSV 文件以写入内容
    with open(merged_file_path, 'w', newline='', encoding='utf-8') as merged_file:
        csv_writer = csv.writer(merged_file)
        for _, file_name in files:
            file_path = os.path.join(source_dir, file_name)
            # 打开当前文件以读取内容
            with open(file_path, 'r', encoding='utf-8') as current_file:
                reader = csv.reader(current_file)
                for row in reader:
                    csv_writer.writerow(row)

print("文件合并完成！")