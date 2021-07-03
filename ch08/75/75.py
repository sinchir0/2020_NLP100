# 75. 損失と正解率のプロット

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

def calc_acc(y_pred_prob, y_true) -> float:
    '''予測のtensorの正解のtensorを用いて、正解率を計算する'''
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(y_pred_prob, 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc

def make_graph(value_dict: dict, value_name: str) -> None:
    '''value_dictに関するgraphを生成し、保存する。'''
    for phase in ['train','test']:
        plt.plot(value_dict[phase],label=phase)
    plt.xlabel('epoch')
    plt.ylabel(value_name)
    plt.title(f'{value_name} per epoch')
    plt.legend()
    plt.savefig(f'{value_name}.png')
    # plt.showしないとだめ？
    plt.show()

if __name__ == "__main__":

    # 学習データの読み込み
    train_x = torch.tensor(
        np.load('../70/train_vector.npy'),
        requires_grad=True
        )

    train_y = torch.tensor(
        np.load('../70/train_label.npy')
        )

    # 評価データの読み込み
    test_x = torch.tensor(
        np.load('../70/test_vector.npy'),
        requires_grad=True
        )

    test_y = torch.tensor(
        np.load('../70/test_label.npy')
        )

    # modelの設定
    net = nn.Linear(300, 4)

    # loss, optimizerの設定
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # parameterの更新    
    for step in range(100):
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
        test_y_pred_prob = net(test_x)

        # 検証データの損失の計算
        test_loss = loss(test_y_pred_prob, test_y)
        # 検証データでの損失の保存
        test_losses.append(test_loss.data)

        # 検証データでの正解率の計算
        test_acc = calc_acc(test_y_pred_prob, test_y)
        # 検証データでの正解率の保存
        test_accs.append(test_acc)

    # グラフへのプロット
    losses = {
        'train': train_losses,
        'test': test_losses
        }

    accs = {
        'train': train_accs,
        'test': test_accs
    }

    make_graph(losses, 'losses')
    make_graph(accs, 'accs')