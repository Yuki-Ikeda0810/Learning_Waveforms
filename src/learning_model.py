#!/usr/bin/env python
# coding: utf-8

import torch

# 畳み込みニューラルネットワークの定義
import torch.nn as nn
import torch.nn.functional as F


########################################################################
# ニューラルネットワークの定義

class Net(nn.Module):

    # Netクラスのコンストラクタ
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, 25)           # 1次元の畳み込み
        self.conv2 = nn.Conv1d(4, 8, 27)           # 1次元の畳み込み
        self.conv3 = nn.Conv1d(8, 12, 25)          # 1次元の畳み込み
        self.conv4 = nn.Conv1d(12, 16, 27)         # 1次元の畳み込み
        self.conv5 = nn.Conv1d(16, 20, 28)         # 1次元の畳み込み
        self.pool = nn.MaxPool1d(4)                # プーリング
        self.fc1 = nn.Linear(20 * 38, 120)         # 全結合(線型変換)
        self.fc2 = nn.Linear(120, 64)              # 全結合(線型変換)
        self.fc3 = nn.Linear(64, 1)                # 全結合(線型変換)
        self.sigmoid = nn.Sigmoid()                # 活性化関数(シグモイド)

    # 順伝播
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))       # 畳み込み -> 活性化(ReLU) -> プーリング
        x = self.pool(F.relu(self.conv2(x)))       # 畳み込み -> 活性化(ReLU) -> プーリング
        x = self.pool(F.relu(self.conv3(x)))       # 畳み込み -> 活性化(ReLU) -> プーリング
        x = self.pool(F.relu(self.conv4(x)))       # 畳み込み -> 活性化(ReLU) -> プーリング
        x = self.pool(F.relu(self.conv5(x)))       # 畳み込み -> 活性化(ReLU) -> プーリング
        x = x.view(-1, self.num_flat_features(x))  # 行列の形状を変換
        x = F.relu(self.fc1(x))                    # 全結合(線型変換) -> 活性化(ReLU)
        x = F.relu(self.fc2(x))                    # 全結合(線型変換) -> 活性化(ReLU)
        x = self.sigmoid(self.fc3(x))              # 全結合(線型変換) -> 活性化(シグモイド)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features