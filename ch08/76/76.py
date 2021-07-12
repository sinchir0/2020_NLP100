# 75. 損失と正解率のプロット

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def calc_acc(y_pred_prob, y_true) -> float:
    """予測のtensorの正解のtensorを用いて、正解率を計算する"""
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(y_pred_prob, 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc


if __name__ == "__main__":

    # 学習データの読み込み
    train_x = torch.tensor(np.load("../70/train_vector.npy"), requires_grad=True)

    train_y = torch.tensor(np.load("../70/train_label.npy"))

    # 検証データの読み込み
    valid_x = torch.tensor(np.load("../70/valid_vector.npy"), requires_grad=True)

    valid_y = torch.tensor(np.load("../70/valid_label.npy"))

    # modelの設定
    net = nn.Linear(300, 4)

    # loss, optimizerの設定
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # parameterの更新
    for epoch in range(100):
        optimizer.zero_grad()

        train_y_pred_prob = net(train_x)

        # 訓練データでの損失の計算
        train_loss = loss(train_y_pred_prob, train_y)
        train_loss.backward()

        optimizer.step()

        # 訓練データでの損失の保存
        train_losses.append(train_loss.data)

        # 訓練データでの正解率の計算
        train_acc = calc_acc(train_y_pred_prob, train_y)
        # 訓練データでの正解率の保存
        train_accs.append(train_acc)

        # 検証データに対する予測
        valid_y_pred_prob = net(valid_x)

        # 検証データの損失の計算
        valid_loss = loss(valid_y_pred_prob, valid_y)
        # 検証データでの損失の保存
        valid_losses.append(valid_loss.data)

        # 検証データでの正解率の計算
        valid_acc = calc_acc(valid_y_pred_prob, valid_y)
        # 検証データでの正解率の保存
        valid_accs.append(valid_acc)

        # 20epoch毎にチェックポイントを生成
        if epoch % 20 == 0:
            torch.save(net.state_dict(), f"76_net_epoch{epoch}.pth")
            torch.save(optimizer.state_dict(), f"76_optimizer_epoch{epoch}.pth")
