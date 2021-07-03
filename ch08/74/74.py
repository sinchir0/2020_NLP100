# 74. 正解率の計測

import numpy as np
import torch
from torch import nn

if __name__ == "__main__":

    # modelの設定
    net = nn.Linear(300, 4)

    net_path = '../73/73_net.pth'
    net.load_state_dict(torch.load(net_path))

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

    # 学習データに対する予測
    train_pred_prob = net(train_x)
    _, train_pred = torch.max(train_pred_prob, 1)

    # 学習データに対する正解率の計算
    train_correct_num = (train_pred == train_y).sum().item()
    train_size = train_y.size(0)
    train_acc = (train_correct_num / train_size) * 100
    print(f'train acc:{train_acc: .2f}%')
    # train acc: 72.84%

    # 評価データに対する予測
    test_pred_prob = net(test_x)
    _, test_pred = torch.max(test_pred_prob, 1)

    # 評価データに対する正解率の計算
    test_correct_num = (test_pred == test_y).sum().item()
    test_size = test_y.size(0)
    test_acc = (test_correct_num / test_size) * 100
    print(f'test acc:{test_acc: .2f}%')
    # test acc: 68.29%