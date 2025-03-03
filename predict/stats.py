import os
import pandas as pd
import matplotlib.pyplot as plt


# 定义统计不同误差范围文件数量的函数
def count_error_ranges(max_errors):
    count_gt_10 = 0
    count_5_to_10 = 0
    count_lt_3 = 0
    count_3_to_5 = 0

    for error in max_errors:
        if error > 10:
            count_gt_10 += 1
        elif 5 < error <= 10:
            count_5_to_10 += 1
        elif error < 3:
            count_lt_3 += 1
        elif 3 <= error <= 5:
            count_3_to_5 += 1

    return count_gt_10, count_5_to_10, count_lt_3, count_3_to_5


# 定义处理文件并绘图的主函数
def process_files(data_dir):
    max_errors = []
    file_names = []

    # 遍历 data 目录下的所有文件
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 读取文件数据，这里假设文件是以逗号分隔的 CSV 文件，可按需修改
                df = pd.read_csv(file_path)
                # 提取第三列数据
                errors = df.iloc[:, 2]
                # 找出最大误差值
                max_error = abs(errors.mean())
                max_errors.append(max_error)
                file_names.append(file)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")

    # 绘制每个文件的最大误差值图
    plt.figure(figsize=(10, 6))
    plt.bar(file_names, max_errors)
    plt.xlabel('文件名')
    plt.ylabel('最大误差 (%)')
    plt.title('每个文件的最大误差')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # 统计不同误差范围的文件数量
    count_gt_10, count_5_to_10, count_lt_3, count_3_to_5 = count_error_ranges(max_errors)

    # 准备绘制不同误差范围文件数量的柱状图
    labels = ['> 10%', '5% - 10%', '3% - 5%', '< 3%']
    counts = [count_gt_10, count_5_to_10, count_3_to_5, count_lt_3]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts)

    # 在柱子上标注数量
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xlabel('err mean')
    plt.ylabel('file number')
    plt.title('file number of err')
    plt.tight_layout()
    # plt.show()
    plt.savefig('./errs_for_each_vm.png')

    print(f"err > 10% file number: {count_gt_10}")
    print(f"err 5% - 10% file number: {count_5_to_10}")
    print(f"err < 3% file number: {count_lt_3}")
    print(f"err 3% - 5% file number: {count_3_to_5}")


# 调用主函数进行处理，需将 'data' 替换为实际的目录路径
data_dir = './predict_out/arima_ret'
process_files(data_dir)
