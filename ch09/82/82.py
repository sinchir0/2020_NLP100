# 82. 確率的勾配降下法による学習
# 確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，問題81で構築したモデルを学習せよ．
# 訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，
# 適当な基準（例えば10エポックなど）で終了させよ．

import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ipdb import set_trace as st
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed=42, use_torch=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(use_torch=True)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx):  # Dataset[index]で返す値を指定
        # Xをintのlistに変換
        X_list = [int(x) for x in self.X[idx].split()]

        # tensorに変換
        inputs = torch.tensor(X_list)
        label = torch.tensor(self.y[idx])

        return inputs, label


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_size):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0  # 0に変換された文字にベクトルを計算しない
        )

        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=True,
        )
        # batch_first=Trueにすると、inputとoutputの型が変わる

        # inputの想定の型を、(L, N, H_in)から(N, L, H_in)に変更する
        # L: Sequence Length
        # N: Batch Size
        # H_in: input size

        # outputの想定の型を、(L, N, D * H_out)から(N, L, D * H_out)に変更する
        # L: Sequence Length
        # N: Batch Size
        # D: 2 if bidirectional=True otherwise 1
        # H_out: hidden size

        self.fc = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_0=None):
        x = self.emb(x)
        x, h_t = self.rnn(x, h_0)
        x = x[:, -1, :]  # sequenceの長さ(現在は10)のうち、一番最後の出力に絞る、やっていいのかこれ？
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train_fn(model, loader, optimizer, criterion, BATCHSIZE, HIDDEN_SIZE) -> float:
    """model, loaderを用いて学習を行い、lossを返す"""

    # 学習モードに設定
    model.train()

    train_running_loss = 0.0

    for dataloader_x, dataloader_y in loader:
        optimizer.zero_grad()

        dataloader_y_pred_prob = model(
            x=dataloader_x
            # x=dataloader_x,
            # h_0=torch.zeros(1 * 1, BATCHSIZE, HIDDEN_SIZE)
        )

        # dataloader_xでの損失の計算
        loss = criterion(dataloader_y_pred_prob, dataloader_y)
        # 勾配の計算
        loss.backward()
        optimizer.step()

        # 訓練データでの損失の平均を計算する
        train_running_loss += loss.item() / len(loader)

    return train_running_loss


def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
    """損失・正解率を計算"""
    # 評価モードに設定
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for dataloader_x, dataloader_y in dataloader:
            # 順伝播
            outputs = model(dataloader_x)

            # 損失計算
            loss += criterion(outputs, dataloader_y).item()

            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(dataloader_x)
            correct += (pred == dataloader_y).sum().item()

    return loss / len(dataset), correct / total


def padding(id_seq: str, max_len: int):
    """id_seqについて、
    max_lenより長い場合はmax_lenまでの長さにする。
    max_lenより短い場合はmax_lenになるように0を追加する。
    """
    id_list = id_seq.split(" ")
    if len(id_list) > max_len:
        id_list = id_list[:max_len]
    else:
        pad_num = max_len - len(id_list)
        for _ in range(pad_num):
            id_list.append("0")
    return " ".join(id_list)


def make_graph(value_dict: dict, value_name: str, bn: int, method: str) -> None:
    """value_dictに関するgraphを生成し、保存する。"""
    for phase in ["train", "test"]:
        plt.plot(value_dict[phase], label=phase)
    plt.xlabel("epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per epoch at bn{bn}")
    plt.legend()
    plt.savefig(f"{method}_{value_name}_bn{bn}.png")
    plt.close()


if __name__ == "__main__":

    # 時間の計測
    start = time.time()

    # データの読み込み
    train = pd.read_pickle("../80/train_title_id.pkl")
    test = pd.read_pickle("../80/test_title_id.pkl")
    test = test.reset_index(drop=True)

    # paddingの実施
    max_len = 10
    train["TITLE"] = train["TITLE"].apply(lambda x: padding(x, max_len))
    test["TITLE"] = test["TITLE"].apply(lambda x: padding(x, max_len))

    # yの変換
    cat_id_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
    train["CATEGORY"] = train["CATEGORY"].map(cat_id_dict)
    test["CATEGORY"] = test["CATEGORY"].map(cat_id_dict)

    # 辞書の読み込み
    with open("../80/word_id_dict.pkl", "rb") as tf:
        word_id_dict = pickle.load(tf)

    N_LETTERS = len(word_id_dict.keys()) + 1  # pad分をplusする。
    EMB_SIZE = 300
    HIDDEN_SIZE = 50
    N_CATEGORIES = 4

    # modelの定義
    model = RNN(
        vocab_size=N_LETTERS,
        emb_dim=EMB_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=N_CATEGORIES,
    )

    # criterion, optimizerの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # datasetの定義
    dataset_train = TextDataset(train["TITLE"], train["CATEGORY"])
    dataset_test = TextDataset(test["TITLE"], test["CATEGORY"])

    # parameterの更新
    # BATCHSIZE = 1
    # BATCHSIZE = 32
    BATCHSIZE = train.shape[0]
    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCHSIZE, shuffle=False, drop_last=True
    )

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # deviceの指定
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    EPOCH = 10
    for epoch in tqdm(range(EPOCH)):

        # 学習
        train_running_loss = train_fn(
            model, dataloader_train, optimizer, criterion, BATCHSIZE, HIDDEN_SIZE
        )
        print(train_running_loss)

        # 損失と正解率の算出
        train_loss, train_acc = calculate_loss_and_accuracy(
            model, dataset_train, device, criterion
        )
        test_loss, test_acc = calculate_loss_and_accuracy(
            model, dataset_test, device, criterion
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 20epoch毎にチェックポイントを生成
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"82_model_epoch{epoch}.pth")
            torch.save(
                optimizer.state_dict(),
                f"82_optimizer_epoch{epoch}.pth",
            )

    # グラフへのプロット
    losses = {"train": train_losses, "test": test_losses}

    accs = {"train": train_accs, "test": test_accs}

    make_graph(losses, "losses", bn=BATCHSIZE, method="rnn")
    make_graph(accs, "accs", bn=BATCHSIZE, method="rnn")

    print(f"train_acc: {train_acc}")
    print(f"test_acc: {test_acc}")

    # 時間の出力
    elapsed_time = time.time() - start
    print(elapsed_time)
