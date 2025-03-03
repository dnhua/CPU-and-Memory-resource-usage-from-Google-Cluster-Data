import os
from pathlib import Path

import numpy as np
import pandas as pd
import random


def truncate_arrays(arrays):
    # 找出所有数组中的最小长度
    min_length = min(len(arr) for arr in arrays)
    truncated_arrays = []
    for arr in arrays:
        # 截取数组到最小长度
        truncated_arr = np.array(arr[:min_length])
        truncated_arrays.append(truncated_arr)
    # 将截取后的数组转换为 numpy 数组
    result = np.array(truncated_arrays)
    return result

# 定义数据目录
data_dir = '../GCD_VMs_new'
out_dir = '../GCD_NODEs'

# 如果目标目录不存在，则创建它
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# 获取数据目录下的所有文件
file_list = os.listdir(data_dir)

# 循环200次，生成200个大文件
for i in range(200):
    print(i)
    # 随机选取20个文件
    selected_files = random.sample(file_list, 20)

    # 存储每个文件的第三列数据
    columns = []

    # 遍历选取的20个文件
    for file in selected_files:
        file_path = os.path.join(data_dir, file)
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file1:
                    lines = file1.readlines()
                    second_column = [float(line.split()[1]) for line in lines]
                    second_column = pd.Series(second_column)
                    columns.append(second_column)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")

    # 将20列数据组合成一个DataFrame
    print(columns)
    # columns = truncate_arrays(columns)
    combined_df = pd.concat(columns, axis=1)


    # 计算前20列的均值乘以20
    combined_df['mean_times_20'] = combined_df.mean(axis=1) * 20

    # 保存新生成的大文件
    output_file_path = Path(out_dir)/f'output_{i + 1}.csv'
    print(output_file_path)
    combined_df.to_csv(output_file_path, index=False)
    print(f"已生成文件: {output_file_path}")

# print(123)