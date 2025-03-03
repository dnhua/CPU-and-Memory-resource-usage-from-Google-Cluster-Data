import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义 result 目录的路径
result_dir = './predict_out/arima_ret'

# 初始化一个空列表来存储所有的准确率数据
accuracies = []

# 遍历 result 目录下的所有文件
for filename in os.listdir(result_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(result_dir, filename)
        try:
            # 读取文件，假设数据以空格或逗号分隔
            data = pd.read_csv(file_path)
            # 提取第三列的误差率数据
            error_rates = data.iloc[:, 2]
            # 计算准确率
            accuracy = 100 - error_rates
            # 将准确率数据添加到列表中
            accuracies.extend(accuracy.tolist())
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# 将准确率数据转换为 numpy 数组
accuracies = np.array(accuracies)

# 单独统计每个区间的数量
below_80 = np.sum((accuracies < 80))
between_80_85 = np.sum((accuracies >= 80) & (accuracies < 85))
between_85_90 = np.sum((accuracies >= 85) & (accuracies < 90))
between_90_95 = np.sum((accuracies >= 90) & (accuracies < 95))
between_95_97 = np.sum((accuracies >= 95) & (accuracies < 97))
above_97 = np.sum(accuracies >= 97)

# 定义区间标签和对应的数量
intervals = ['<80%', '80%-85%', '85%-90%', '90%-95%', '95%-97%', '>=97%']
counts = [below_80, between_80_85, between_85_90, between_90_95, between_95_97, above_97]
print(counts)

# 绘制柱状图
bars = plt.bar(intervals, counts)
# 在柱子上标注数量
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')
plt.xlabel('Accuracy Intervals')
plt.ylabel('Number of Samples')
plt.title('Distribution of Prediction Accuracy')
plt.grid(True)
# plt.show()
plt.savefig('./errs_for_each_point.png')
