#!/usr/bin/env python
# coding: utf-8

# Pytorch
import torch

# 畳み込みニューラルネットワークの定義
import torch.nn as nn

# 定義した畳み込みニューラルネットワークを読み込む
import test10_model

# 損失関数と最適化アルゴリズムを定義する
import torch.optim as optim

# 行列計算
import numpy as np

# 高速フーリエ変換(FFT)のライブラリ
from scipy.fftpack import fft, fftfreq

# 並列処理のライブラリ(windowsで必要)
from multiprocessing import freeze_support

# path指定用
import os

# グラフ表示用のライブラリ
import matplotlib.pyplot as plt

########################################################################
# 初期設定

# deviceの設定(GPUを使う場合)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 現在のパスを取得
if len(os.path.dirname(__file__)) == 0:
    current_path = os.getcwd()
else :
    current_path = os.path.dirname(__file__)

# 入力データのpathの設定
path = current_path + "\\data\\"

# 各変数の定義
batch_size = 1

epoch_size = 100

train_data = []
train_label = []

test_data = []
test_label = []

loss_array = []
plt.figure()

########################################################################
# データの読み込み

print("Loading Dateset......")

for i in range(28):
    # トレーニング用データローダ(正常波形)
    train_data.append(np.abs(fft(np.loadtxt(path + "nomal\\{}.csv".format(i)))))
    train_label.append(0)

    # トレーニング用データローダ(異常波形)
    if(i < 14):
        train_data.append(np.abs(fft(np.loadtxt(path + "1cm\\{}.csv".format(i)))))
        train_label.append(1)
        train_data.append(np.abs(fft(np.loadtxt(path + "3cm\\{}.csv".format(i)))))
        train_label.append(1)

train_label = torch.from_numpy(np.array(train_label)).float()
train_label = train_label.view(-1, 1)

train_data = torch.from_numpy(np.array(train_data)).float()
train_data = train_data.view(-1, 1, len(train_data[0]))

print(train_data.size())

traindataset = torch.utils.data.TensorDataset(train_data, train_label)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size)

print("Loading Complete.")


########################################################################
# メイン関数

if __name__ == "__main__":
    freeze_support()        #並列処理(windowsで必要)

    net = test10_model.Net()

    net.to(device)          # GPUを使う場合
    print(device)           # GPUを使う場合

    criterion = nn.MSELoss()                                # 損失の計算
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # パラメータの更新


    ########################################################################
    # トレーニングデータで学習する

    for epoch in range(epoch_size):  # エポック数(ループする回数)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # トレーニングデータを取得する
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)      # GPUを使う場合

            # 勾配を初期化する
            optimizer.zero_grad()

            # ニューラルネットワークにデータを通し、順伝播を計算する
            outputs = net(inputs)

            # 誤差の計算
            loss = criterion(outputs, labels)

            # 逆伝播の計算
            loss.backward()

            # 重みの計算
            optimizer.step()

            # 状態を表示する
            running_loss += loss.item()

        print('[%02d, %5d] loss: %.10f' % (epoch + 1, i + 1, running_loss/len(trainloader)))
        loss_array.append((running_loss/len(trainloader)))
        running_loss = 0.0

        plt.clf()
        plt.plot(loss_array, color='blue', label='running loss')
        plt.ylim(0, loss_array[0] + 0.05)
        plt.xlim(1, epoch_size)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.001)

        # 学習したモデルの保存
        # print('Save Training Model')
        model_path = current_path + "\\model\model_{}.pth".format(epoch + 1)
        torch.save(net.state_dict(), model_path)

    print('Finished Training')
    _ = input('press the Enter to Exit')