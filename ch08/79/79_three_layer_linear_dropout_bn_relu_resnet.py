# kaggle.com/qiaoshiji/resnet-deep

# 79. 多層ニューラルネットワーク
# 問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        self.fc1 = nn.Linear(300, 150, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 150, bias=True)
        self.dropout2 = nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm1d(150)
        self.fc3 = nn.Linear(300, 4, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.fc3(x2)
        x3 = self.softmax(x3)

        return x3


def train_fn(model, loader, optimizer, loss) -> Union[float, float]:
    """model, loaderを用いて学習を行い、lossを返す"""
    train_running_loss = 0.0
    valid_running_loss = 0.0

    for dataloader_x, dataloader_y in loader:
        optimizer.zero_grad()

        dataloader_y_pred_prob = model(dataloader_x)

        # dataset_xでの損失の計算
        dataloader_loss = loss(dataloader_y_pred_prob, dataloader_y)
        dataloader_loss.backward()

        # 訓練データ、検証データでの損失の平均を計算する
        train_running_loss += dataloader_loss.item() / len(loader)
        valid_running_loss += loss(model(valid_x), valid_y).item() / len(loader)

        optimizer.step()

    return train_running_loss, valid_running_loss


def calc_acc(model, train_x, y_true) -> float:
    """modelと学習データ、正解データを用いて、正解率を計算する"""
    # 最も正解率の高い予測確率を正解ラベルとする。
    _, y_pred = torch.max(model(train_x), 1)

    # 学習データに対する正解率の計算
    correct_num = (y_pred == y_true).sum().item()
    total_size = y_true.size(0)
    acc = (correct_num / total_size) * 100
    return acc


def make_graph(value_dict: dict, value_name: str, method: str) -> None:
    """value_dictに関するgraphを生成し、保存する。"""
    for phase in ["train", "valid"]:
        plt.plot(value_dict[phase], label=phase)
    plt.xlabel("epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per epoch")
    plt.legend()
    plt.savefig(f"{method}_{value_name}.png")
    plt.close()


if __name__ == "__main__":

    METHOD = "three_layer_linear_dropout_bn"

    if not torch.cuda.is_available():
        print("No cuda")

    PATH = ".."

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # 学習データの読み込み
    train_x = torch.tensor(
        np.load(f"{PATH}/70/train_vector.npy"), requires_grad=True
    ).to(device)
    train_y = torch.tensor(np.load(f"{PATH}/70/train_label.npy")).to(device)

    # 評価データの読み込み
    valid_x = torch.tensor(
        np.load(f"{PATH}/70/valid_vector.npy"), requires_grad=True
    ).to(device)
    valid_y = torch.tensor(np.load(f"{PATH}/70/valid_label.npy")).to(device)

    # modelの設定
    model = Net(in_shape=train_x.shape[1], out_shape=4).to(device)

    # loss, optimizerの設定
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # DataLoaderの構築
    dataset = TextDataset(train_x, train_y)

    # parameterの更新
    BATCHSIZE = 32
    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    EPOCH = 10
    for epoch in tqdm(range(EPOCH)):

        # 学習
        train_running_loss, valid_running_loss = train_fn(
            model, loader, optimizer, loss
        )

        # 訓練データでの損失の保存
        train_losses.append(train_running_loss)

        # 訓練データでの正解率の計算
        train_acc = calc_acc(model, train_x, train_y)
        # 訓練データでの正解率の保存
        train_accs.append(train_acc)

        # 検証データでの損失の保存
        valid_losses.append(valid_running_loss)

        # 検証データでの正解率の計算
        valid_acc = calc_acc(model, valid_x, valid_y)
        # 検証データでの正解率の保存
        valid_accs.append(valid_acc)

        # 20epoch毎にチェックポイントを生成
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"79_model_bs_epoch{epoch}.pth")
            torch.save(
                optimizer.state_dict(),
                f"79_optimizer_epoch{epoch}.pth",
            )

    # グラフへのプロット
    losses = {"train": train_losses, "valid": valid_losses}

    accs = {"train": train_accs, "valid": valid_accs}

    make_graph(losses, "losses", METHOD)
    make_graph(accs, "accs", METHOD)

    print(f"train_acc: {train_acc}")
    print(f"valid_acc: {valid_acc}")
    # train_acc: 84.21101949025487
    # valid_acc: 85.83208395802099
