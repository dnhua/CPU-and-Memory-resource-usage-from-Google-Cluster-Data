import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

def pad_arrays(arrays):
    # 找出所有数组中的最大长度
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = []
    for arr in arrays:
        # 计算需要填充的零的数量
        padding_length = max_length - len(arr)
        # 如果需要填充，使用 np.pad 函数在数组末尾填充零
        if padding_length > 0:
            padded_arr = np.pad(arr, (0, padding_length), 'constant')
        else:
            padded_arr = np.array(arr)
        padded_arrays.append(padded_arr)
    # 将填充后的数组转换为 numpy 数组
    result = np.array(padded_arrays)
    return result

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

def kmeans(train_dir, out_dir, prefix):
    # 读取train目录下每个文件的第二列数据
    data = []
    filenames = []
    for filename in os.listdir(train_dir):
        train_file_path = os.path.join(train_dir, filename)
        if os.path.isfile(train_file_path):
            with open(train_file_path, 'r') as file:
                lines = file.readlines()
                second_column = [float(line.split()[1]) for line in lines]
                data.append(second_column)
                filenames.append(filename)

    # 将数据转换为numpy数组
    # data = pad_arrays(data)
    data = truncate_arrays(data)

    # 使用KMeans算法进行聚类，聚成五类
    n_clusters = 5
    mode = KMeans(n_clusters=n_clusters, random_state=42)
    labels = mode.fit_predict(data)

    # 创建目标train聚类结果目录
    for i in range(n_clusters):
        target_train_dir = Path(out_dir)/f"{prefix}{i}"
        if not os.path.exists(target_train_dir):
            os.makedirs(target_train_dir)

    # 用于记录每个分类的文件数量
    cluster_counts = [0] * n_clusters

    # 将train和test中的文件拷贝到目标train和test目录
    for i in range(len(filenames)):
        filename = filenames[i]
        label = labels[i]
        train_source_file_path = os.path.join(train_dir, filename)
        target_train_dir = Path(out_dir)/f"{prefix}{label}"
        target_train_file_path = os.path.join(target_train_dir, filename)
        shutil.copy2(train_source_file_path, target_train_file_path)
        cluster_counts[label] += 1

    # 打印每个分类的文件数量
    for i in range(n_clusters):
        print(f"分类 {i} 中有 {cluster_counts[i]} 个文件。")