#!/usr/bin/env python
# coding: utf-8

import torch

# 畳み込みニューラルネットワークの定義
import torch.nn as nn

# 定義した畳み込みニューラルネットワークを読み込む
import learning_model

# 行列計算
import numpy as np

# 高速フーリエ変換(FFT)のライブラリ
from scipy.fftpack import fft, fftfreq

# 並列処理のライブラリ(windowsで必要)
from multiprocessing import freeze_support

# path指定用
import os

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

train_data = []
train_label = []

test_data = []
test_label = []

########################################################################
# データの読み込み

print("Loading Dateset......")

# テスト用データローダ(正常波形)
for i in range(28, 40):
    test_data.append(np.abs(fft(np.loadtxt(path + "nomal\\{}.csv".format(i)))))
    test_label.append(0)

# テスト用データローダ(異常波形)
for i in range(15, 20):
    test_data.append(np.abs(fft(np.loadtxt(path + "1cm\\{}.csv".format(i)))))
    test_label.append(1)
    test_data.append(np.abs(fft(np.loadtxt(path + "3cm\\{}.csv".format(i)))))
    test_label.append(1)

test_label = torch.from_numpy(np.array(test_label)).float()
test_label = test_label.view(-1, 1)

test_data = torch.from_numpy(np.array(test_data)).float()
test_data = test_data.view(-1, 1, 48000)

testdataset = torch.utils.data.TensorDataset(test_data, test_label)
testloader = torch.utils.data.DataLoader(testdataset, batch_size)

print("Loading complete.")


########################################################################
# メイン関数

if __name__ == "__main__":
    freeze_support()        #並列処理(windowsで必要)

    net = test10_model.Net()

    net.to(device)          # GPUを使う場合
    print(device)           # GPUを使う場合

    # 学習したモデルの読み込み
    print('Load Training Model')
    model_path = current_path + "\\model\model_100.pth"
    net.load_state_dict(torch.load(model_path))

    ########################################################################
    # テスト用データでテストする
    net.eval()
    dataiter = iter(testloader)
    waves, labels = dataiter.next()

    waves, labels = waves.to(device), labels.to(device)            # GPUを使う場合
    print(labels)
    print('正解ラベル: ', ' '.join('%5s' % labels[j] for j in range(1)))

    outputs = net(waves)
    if outputs[0][0].item() < 0.5:
        print("output: ", 0)
    else:
        print("output: ", 1)
    print(outputs[0,:])

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            waves, labels = data
            waves, labels = waves.to(device), labels.to(device)    # GPUを使う場合
            outputs = net(waves)
            out = 0

            if outputs[0][0].item() < 0.5:
                #print("output: ", 0)
                out = 0
            else:
                out = 1
                #print("output: ", 1)

            print("prediction: %d, answer: %d, probability :%f" % (out, labels[0][0], outputs[0][0].item()))

            if labels[0][0] == out:
                correct += 1
            total += 1
            
    print('Accuracy of the network on the %d test waves: %d %%' % (len(testloader), 100 * correct / total))
