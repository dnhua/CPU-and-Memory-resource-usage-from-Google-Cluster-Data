import os
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_model(data_dir, out_dir, walk_step):
    # 遍历类别目录中的每个文件
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        # 读取文件的第二列数据
        print(file_path)
        data = pd.read_csv(file_path, sep='\s+', usecols=[1], header=None, dtype=np.float64)
        data = data.values.flatten()
        # print(data)

        predictions = []
        actuals = []

        # 从第 walk_step+1 个数据点开始，使用前 walk_step 个数据点进行预测
        for i in range(walk_step, len(data)):
            train_data = data[i - walk_step:i]
            actual = data[i]
            # print(train_data)

            try:
                # 训练 ARIMA 模型，这里假设 p=1, d=1, q=1，可以根据实际情况调整
                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()

                # 进行预测
                prediction = model_fit.forecast(steps=1)[0]
                predictions.append(prediction)
                actuals.append(actual)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
        # Save actual/predictions/err to file.
        print(predictions)
        print(actuals)
        err = (np.array(actuals) - np.array(predictions))/np.array(actuals)*100
        print(err)
        save_ret = {
            'actual': actuals,
            'prediction': predictions,
            'err%': err
        }
        save_df = pd.DataFrame(save_ret)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_df.to_csv(Path(out_dir)/f'ret_{filename}', index=False)

    print("所有类别文件的 ARIMA 训练、评估和绘图完成。")