# 89. 事前学習済み言語モデルからの転移学習
# 事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．

# https://qiita.com/yamaru/items/63a342c844cff056a549

import os
import pickle
import random
import time

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ipdb import set_trace as st
from numpy.lib.function_base import kaiser
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AlbertModel, AlbertTokenizer


def seed_everything(seed=42, use_torch=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


class TextDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len, device):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.X)

    def __getitem__(self, idx):  # Dataset[index]で返す値を指定
        text = self.X[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, device=device),
            "mask": torch.tensor(mask, device=device),
            "labels": torch.tensor([self.y[idx]], device=device),
        }


class AlbertClass(nn.Module):
    def __init__(self, drop_rate, output_size):
        super().__init__()
        self.model = AlbertModel.from_pretrained("albert-base-v2")
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ids, mask):
        outputs = self.model(ids, attention_mask=mask)
        _, x = outputs["last_hidden_state"], outputs["pooler_output"]
        x = self.drop(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train_fn(model, dataset, device, optimizer, criterion, BATCH_SIZE) -> float:
    """model, loaderを用いて学習を行い、lossを返す"""

    # dataloaderを生成
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 学習モードに設定
    model.train()

    train_running_loss = 0.0

    for data in loader:
        # デバイスの指定
        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        labels = data["labels"].to(device)

        # labelsの次元を(BATCH_SIZE, 1)から(BATCH_SIZE,)に変更
        labels = labels.squeeze(1)

        optimizer.zero_grad()

        outputs = model(ids, mask)
        # dataloader_xでの損失の計算/
        loss = criterion(outputs, labels)
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
        for data in dataloader:
            # 変数の生成、device送り
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            # 順伝播
            outputs = model(ids, mask)

            # labelsの次元を(BATCH_SIZE, 1)から(BATCH_SIZE,)に変更
            labels = labels.squeeze(1)

            # 損失計算
            loss += criterion(outputs, labels).item()

            # 正解率計算
            pred = torch.argmax(outputs, dim=-1)
            total += len(labels)
            correct += (pred == labels).sum().item()

    return loss / len(dataset), correct / total


def make_graph(value_dict: dict, value_name: str, bn: int, method: str) -> None:
    """value_dictに関するgraphを生成し、保存する。"""
    for phase in ["train", "test"]:
        plt.plot(value_dict[phase], label=phase)
    plt.xlabel("epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per epoch at bn{bn}")
    plt.legend()
    plt.savefig(f"{PATH}/ch09/89/{method}_{value_name}_bn{bn}.png")
    plt.close()


if __name__ == "__main__":

    DEBUG = True
    if DEBUG:
        print("DEBUG mode")

    METHOD = "albert"

    # 時間の計測開始
    start_time = time.time()

    # seedの固定
    seed_everything(use_torch=True)

    # Colab
    # PATH = '/content/drive/MyDrive/NLP100/ch09'
    # local
    PATH = "../../"

    # deviceの指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Use {device}")

    # データの読み込み
    train = pd.read_csv(f"{PATH}/ch06/50/train.txt", sep="\t", index_col=0)
    test = pd.read_csv(f"{PATH}/ch06/50/test.txt", sep="\t", index_col=0)

    # indexを再設定
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # 計算の短縮
    if DEBUG:
        train = train.sample(1000).reset_index(drop=True)
        test = test.sample(1000).reset_index(drop=True)

    # 正解データの生成
    cat_id_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
    y_train = train["CATEGORY"].map(cat_id_dict)
    y_test = test["CATEGORY"].map(cat_id_dict)

    # tokenizerの読み込み
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    max_len = 10

    # datasetの作成
    dataset_train = TextDataset(train["TITLE"], y_train, tokenizer, max_len, device)
    dataset_test = TextDataset(test["TITLE"], y_test, tokenizer, max_len, device)
    # {'input_ids': [101, 2885, 6561, 24514, 2391, 2006, 8169, 2586, 102, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    # BERTでは、変換の過程で元の文の文頭と文末に特殊区切り文字である[CLS]と[SEP]がそれぞれ挿入されるため、それらも101と102として系列に含まれています。0はパディングを表します。

    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    if DEBUG:
        NUM_EPOCHS = 2
    LEARNING_RATE = 2e-5

    # モデルの定義
    model = AlbertClass(DROP_RATE, OUTPUT_SIZE).to(device)

    # criterion, optimizerの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        # 学習
        train_running_loss = train_fn(
            model, dataset_train, device, optimizer, criterion, BATCH_SIZE
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
            torch.save(model.state_dict(), f"89_{METHOD}_epoch{epoch}.pth")
            torch.save(
                optimizer.state_dict(),
                f"89_{METHOD}_optimizer_epoch{epoch}.pth",
            )

    # グラフへのプロット
    losses = {"train": train_losses, "test": test_losses}

    accs = {"train": train_accs, "test": test_accs}

    make_graph(losses, "losses", bn=BATCH_SIZE, method=METHOD)
    make_graph(accs, "accs", bn=BATCH_SIZE, method=METHOD)

    print(f"train_acc: {train_acc: .4f}")
    print(f"test_acc: {test_acc: .4f}")

    # 計測終了
    elapsed_time = time.time() - start_time
    print(f"elapsed_time:{elapsed_time:.0f}[sec]")
