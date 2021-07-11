# 77.ミニバッチ化

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.fc = nn.Linear(300, 4, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


def calc_acc(net, train_x, y_true) -> float:
    """modelと学習データ、正解データを用いて、正解率を計算する"""
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(net(train_x), 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc


if __name__ == "__main__":

    # 学習データの読み込み
    train_x = torch.tensor(np.load("../70/train_vector.npy"), requires_grad=True)
    train_y = torch.tensor(np.load("../70/train_label.npy"))

    # 評価データの読み込み
    valid_x = torch.tensor(np.load("../70/valid_vector.npy"), requires_grad=True)
    valid_y = torch.tensor(np.load("../70/valid_label.npy"))

    # modelの設定
    net = Net(in_shape=train_x.shape[1], out_shape=4)

    # loss, optimizerの設定
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # DataLoaderの構築
    dataset = TextDataset(train_x, train_y)

    # parameterの更新
    batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
    for batchsize in batchsizes:
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []

        for epoch in range(1):

            start = time.time()

            train_running_loss = 0.0
            valid_running_loss = 0.0

            for dataloader_x, dataloader_y in loader:
                """netの重みの学習をbatchsize単位で行う"""

                optimizer.zero_grad()

                dataloader_y_pred_prob = net(dataloader_x)

                # dataset_xでの損失の計算
                dataloader_loss = loss(dataloader_y_pred_prob, dataloader_y)
                dataloader_loss.backward()

                # 訓練データ、検証データでの損失の平均を計算する
                train_running_loss += dataloader_loss.item()
                valid_running_loss += loss(net(valid_x), valid_y).item()

                optimizer.step()

            # 訓練データでの損失の保存
            train_losses.append(train_running_loss)

            # 訓練データでの正解率の計算
            train_acc = calc_acc(net, train_x, train_y)
            # 訓練データでの正解率の保存
            train_accs.append(train_acc)

            # 検証データでの損失の保存
            valid_losses.append(valid_running_loss)

            # 検証データでの正解率の計算
            valid_acc = calc_acc(net, valid_x, valid_y)
            # 検証データでの正解率の保存
            valid_accs.append(valid_acc)

            # 20epoch毎にチェックポイントを生成
            if epoch % 20 == 0:
                torch.save(net.state_dict(), f"77_net_bs{batchsize}_epoch{epoch}.pth")
                torch.save(
                    optimizer.state_dict(),
                    f"77_optimizer_bs{batchsize}_epoch{epoch}.pth",
                )

            # 経過した時間を取得
            elapsed_time = time.time() - start
            print(f"batchsize{batchsize} time:{elapsed_time: .2f}")

            # batchsize1 time: 20.91
            # batchsize2 time: 26.53
            # batchsize4 time: 21.08
            # batchsize8 time: 18.27
            # batchsize16 time: 16.90
            # batchsize32 time: 16.23
            # batchsize64 time: 15.85
            # batchsize128 time: 15.68
