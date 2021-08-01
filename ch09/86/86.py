# 86. 畳み込みニューラルネットワーク (CNN)

import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.function_base import kaiser
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
    def __init__(self, X, y, device):
        self.X = X
        self.y = y
        self.device = device

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx):  # Dataset[index]で返す値を指定
        # Xをintのlistに変換
        X_list = [int(x) for x in self.X[idx].split()]

        # tensorに変換
        inputs = torch.tensor(X_list, device=self.device)
        label = torch.tensor(self.y[idx], device=self.device)

        return inputs, label


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_size, word_id_dict):
        super().__init__()

        # # embedding層の初期化
        # model = gensim.models.KeyedVectors.load_word2vec_format('../../ch07/60/GoogleNews-vectors-negative300.bin', binary=True)
        # weight = torch.zeros(len(word_id_dict)+1, 300) # 1はPADの分
        # for word, idx in word_id_dict.items():
        #     if word in model.vocab.keys():
        #         weight[idx] = torch.tensor(model[word])

        # # pretrainする場合のemb
        # # vocab_size, emb_dimは、GoogleNews-vectors-negative300が3000000語彙で300次元だから指定しなくて良い
        # self.emb = nn.Embedding.from_pretrained(
        #     weight,
        #     padding_idx=0 # 0に変換された文字にベクトルを計算しない
        #     )

        # default
        self.emb = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0  # 0に変換された文字にベクトルを計算しない
        )

        # self.rnn = nn.RNN(
        #     input_size=emb_dim,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     nonlinearity='tanh',
        #     bias=True,
        #     batch_first=True,
        #     bidirectional=True
        #     )

        self.conv1d = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            padding=0,  # 4方向にpaddingを行う
            padding_mode="zeros",
        )
        # input size: (N, C_in, L)
        # N=batch size
        # C_m=number of channels
        # L=length of signal sequence
        # output size: (N, C_out, L_out)

        self.pool1d = nn.MaxPool1d(
            kernel_size=8  # conv1dにて300→50にすると、Sequential lengthは10→8になって返ってくる。この8を1にしたいので、全体でpooling
        )
        # input size: (N, C, L)
        # N=batch size
        # C_m=number of channels
        # L=length of signal sequence
        # output size: (N, C, L_out)

        self.fc = nn.Linear(
            in_features=hidden_size, out_features=output_size, bias=True
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h_0=None):
        x = self.emb(x)
        x = x.permute(
            0, 2, 1
        )  # 2次元目の数値を1次元目と入れ替え、1次元目にemb dim,2次元目にSequential lengthを持ってくる
        x = self.conv1d(x)
        x = self.pool1d(x)
        x = x[:, :, -1]  # (N, C, L_out)の次元L_outをなくす
        x = self.fc(x)
        x = self.softmax(x)
        return x
        # UserWarning: Named tensors and・・・で謎Warniningが出るけど、気にしなくて良いっぽい
        # https://github.com/pytorch/pytorch/pull/60059


def train_fn(
    model, loader, device, optimizer, criterion, BATCHSIZE, HIDDEN_SIZE
) -> float:
    """model, loaderを用いて学習を行い、lossを返す"""
    # 学習モードに設定
    model.train()

    train_running_loss = 0.0

    for dataloader_x, dataloader_y in loader:
        dataloader_x.to(device)
        dataloader_y.to(device)

        # optimizer.zero_grad()

        dataloader_y_pred_prob = model(
            x=dataloader_x,
            h_0=torch.zeros(2 * 1, BATCHSIZE, HIDDEN_SIZE).to(
                device
            ),  # D * num_layer, bidirectionnalの場合はD=2
        )

        # dataloader_xでの損失の計算/
        loss = criterion(dataloader_y_pred_prob, dataloader_y)
        # 勾配の計算
        loss.backward()
        # optimizer.step()

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

    # Colab
    # PATH = '/content/drive/MyDrive/NLP100/ch09'
    # local
    PATH = ".."

    # データの読み込み
    train = pd.read_csv(f"{PATH}/80/train_title_id.csv")
    test = pd.read_csv(f"{PATH}/80/test_title_id.csv")
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
    with open(f"{PATH}/80/word_id_dict.pkl", "rb") as tf:
        word_id_dict = pickle.load(tf)

    N_LETTERS = len(word_id_dict.keys()) + 1  # pad分をplusする。
    EMB_SIZE = 300
    HIDDEN_SIZE = 50
    N_CATEGORIES = 4

    # deviceの指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use {device}")

    # modelの定義
    model = CNN(
        vocab_size=N_LETTERS,
        emb_dim=EMB_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=N_CATEGORIES,
        word_id_dict=word_id_dict,
    ).to(device)

    # criterion, optimizerの設定
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer = None

    # datasetの定義
    dataset_train = TextDataset(train["TITLE"], train["CATEGORY"], device)
    dataset_test = TextDataset(test["TITLE"], test["CATEGORY"], device)

    # parameterの更新
    BATCHSIZE = 32
    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCHSIZE, shuffle=True, drop_last=True
    )

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    EPOCH = 10
    for epoch in tqdm(range(EPOCH)):

        # 学習
        train_running_loss = train_fn(
            model,
            dataloader_train,
            device,
            optimizer,
            criterion,
            BATCHSIZE,
            HIDDEN_SIZE,
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
            torch.save(model.state_dict(), f"86_model_epoch{epoch}.pth")
            # torch.save(
            #     optimizer.state_dict(),
            #     f"86_optimizer_epoch{epoch}.pth",
            # )

    # グラフへのプロット
    losses = {"train": train_losses, "test": test_losses}

    accs = {"train": train_accs, "test": test_accs}

    make_graph(losses, "losses", bn=BATCHSIZE, method="cnn")
    make_graph(accs, "accs", bn=BATCHSIZE, method="cnn")

    print(f"train_acc: {train_acc: .4f}")
    print(f"test_acc: {test_acc: .4f}")
