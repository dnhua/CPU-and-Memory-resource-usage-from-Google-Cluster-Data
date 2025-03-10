from pathlib import Path
from cluster import kmeans
from predict.arima import arima_model

kmeans_train_dir = '../GCD_VMs_new'
kmeans_target_dir = './predict_out/kmeans_out'
prefix = 'kmeans_'
kmeans_dir = './predict_out/kmeans_out'
# arima_out = './predict_out/kmeans_arima_out'

kmeans.kmeans(kmeans_train_dir, kmeans_target_dir, prefix)
for i in range(9,15):
    print(i)
    # 按照每个类别进行arima预测
    for j in range(5):
        data_dir = Path(kmeans_dir)/f"{prefix}{j}"
        arima_out = f"./predict_out/kmeans_arima_out_{i}"
        arima_model(data_dir, arima_out, i)