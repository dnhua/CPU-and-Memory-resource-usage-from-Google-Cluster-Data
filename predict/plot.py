import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 定义目录路径
directory = './arima_ret'
out_dir = './arima_plot'

Path(out_dir).mkdir(parents=True, exist_ok=True)
# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # 假设文件是 CSV 格式
        file_path = os.path.join(directory, filename)

        # 读取文件
        df = pd.read_csv(file_path)

        # 提取真实数据和预测数据
        true_data = df.iloc[:, 0]
        pred_data = df.iloc[:, 1]

        # 创建一个新的图形
        plt.figure(figsize=(12, 8))

        # 绘制真实数据和预测数据
        plt.plot(true_data, label='real', color='blue')
        plt.plot(pred_data, label='predict', color='red')

        # 设置图形标题和标签
        plt.title(f'{filename} prediction')
        plt.xlabel('time/5min')
        plt.ylabel('usage ration')
        plt.legend()
        plt.ylim(0, 100)

        # 显示网格线
        plt.grid(True)

        # 保存图形
        output_filename = os.path.splitext(filename)[0] + '_comparison.png'
        output_path = os.path.join(out_dir, output_filename)
        plt.savefig(output_path)

        # 关闭图形
        plt.close()

print('所有文件的比较图已生成并保存。')