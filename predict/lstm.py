import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os

# 检查 MPS 可用性
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 准备数据
def prepare_data(data, seq_length):
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

# 划分训练集和测试集
def split_train_test(inputs, targets, test_ratio=0.2):
    test_size = int(len(inputs) * test_ratio)
    train_inputs = inputs[:-test_size]
    train_targets = targets[:-test_size]
    test_inputs = inputs[-test_size:]
    test_targets = targets[-test_size:]
    return train_inputs, train_targets, test_inputs, test_targets

# 训练模型
def train_model(model, train_inputs, train_targets, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs_tensor = torch.from_numpy(train_inputs).float().unsqueeze(2).to(device)
        targets_tensor = torch.from_numpy(train_targets).float().unsqueeze(1).to(device)

        outputs = model(inputs_tensor)
        loss = criterion(outputs, targets_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# 预测并保存结果
def predict_and_save(model, test_inputs, test_targets, output_file):
    predictions = []
    true_values = []
    with torch.no_grad():
        test_inputs_tensor = torch.from_numpy(test_inputs).float().unsqueeze(2).to(device)
        outputs = model(test_inputs_tensor)
        predictions = outputs.cpu().squeeze().numpy()
        true_values = test_targets

    errors = (true_values - predictions)/true_values*100

    results = np.column_stack((true_values, predictions, errors))
    save_ret = {
        'actual': true_values,
        'prediction': predictions,
        'err%': errors
    }
    save_df = pd.DataFrame(save_ret)
    # save_df.to_csv(Path(out_dir) / f'ret_{filename}', index=False)
    np.savetxt(output_file, results, delimiter=',', header='True Value,Predicted Value,Error', comments='')

# 主函数
def main(data_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_size = 1
    hidden_size = 32
    num_layers = 2
    output_size = 1
    seq_length = 10
    num_epochs = 1000
    learning_rate = 0.001
    test_ratio = 0.2

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt') or filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            data = np.loadtxt(file_path, usecols=1)

            inputs, targets = prepare_data(data, seq_length)
            train_inputs, train_targets, test_inputs, test_targets = split_train_test(inputs, targets, test_ratio)

            model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
            trained_model = train_model(model, train_inputs, train_targets, num_epochs, learning_rate)

            output_file = os.path.join(output_folder, f'results_{filename}')
            predict_and_save(trained_model, test_inputs, test_targets, output_file)

if __name__ == '__main__':
    data_folder = '../GCD_VMs_new'  # 数据文件夹路径
    output_folder = './predict_out/lstm_ret'  # 结果保存文件夹路径
    main(data_folder, output_folder)