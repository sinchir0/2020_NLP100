from typing import Tuple

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import texthero as hero
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def train_valid_test_split(data, split_point: Tuple, shuffle=False):
    """pd.DataFrame()をtrain,valid,testに分割する
    Args:
        data: pd.DataFrame()のデータ
        split_point: 分割点
        shuffle: dataをランダムにシャッフルするかどうか
    Returns:
        train,valid,test: ３分割されたdata
    """

    if shuffle:
        data = data.sample(frac=1)

    data_len = len(data)

    first = int(data_len * split_point[0])
    second = int(data_len * split_point[1])
    third = int(data_len * split_point[2])

    train = data[:first]
    valid = data[first : (first + second)]
    test = data[(first + second) : (first + second + third)]

    return train, valid, test


def plot_confusion_matrix(
    y_true,
    y_pred,
    normalize=False,
    title=None,
    labels=None,
    figsize=(8, 12),
    cmap=plt.cm.Blues,
    save_fig=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)

    # labelsを設定している場合はそのlabelのみを使う
    if labels is not None:
        classes = labels

    if normalize:
        # 行方向への正規化
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.figure(figsize=figsize)
    if save_fig:
        fig.savefig(f"{title}.png")
    return ax


def make_predict_by_LR(
    train_y, valid_y, test_y, train_feature, valid_feature, test_feature, C: int
):

    clf = LogisticRegression(solver="liblinear", C=C)
    clf.fit(train_feature, train_y["CATEGORY"])

    train_pred = clf.predict(train_feature)
    valid_pred = clf.predict(valid_feature)
    test_pred = clf.predict(test_feature)

    train_score = accuracy_score(train_y, train_pred)
    valid_score = accuracy_score(valid_y, valid_pred)
    test_score = accuracy_score(test_y, test_pred)

    return [C, train_score, valid_score, test_score]


# 引用：https://github.com/takapy0210/nlp_2020/blob/master/chapter6/ans_51.py
def load_data(data_check=False) -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
<<<<<<< HEAD:src/util.py
        'train': '../ch06/50/train.txt',
        'valid': '../ch06/50/valid.txt',
        'test': '../ch06/50/test.txt',
=======
        "train": "../../ch06/50/train.txt",
        "valid": "../../ch06/50/valid.txt",
        "test": "../../ch06/50/test.txt",
>>>>>>> 16dafcaa09f28883e75f23c09ab0bdb064b2d4e0:src/NLP100/util.py
    }

    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v, sep="\t", index_col=0)

    # データチェック
    if data_check:
        for k in inputs.keys():
            print(k, "---", dfs[k].shape)
            print(dfs[k].head())

    return dfs


def preprocess(text: str) -> str:
    """前処理"""
<<<<<<< HEAD:src/util.py
    clean_text = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace
    ])
=======
    clean_text = hero.clean(
        text,
        pipeline=[
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_stopwords,
        ],
    )
>>>>>>> 16dafcaa09f28883e75f23c09ab0bdb064b2d4e0:src/NLP100/util.py

    return clean_text


class TextFeatureFitTransform:
    """テキストから特徴量を学習し、生成"""

    def __init__(self, method: str) -> None:
        if method == "cntvec":
            self.vec = CountVectorizer()
        elif method == "tfidf":
            self.vec = TfidfVectorizer(ngram_range=(1, 2))
        else:
            raise ValueError("No method")

    def fit(self, input_text) -> None:
        self.vec.fit(input_text)

    def transform(self, input_text) -> pd.DataFrame:
        return self.vec.transform(input_text)
<<<<<<< HEAD:src/util.py

    def vocab(self) -> dict:
        return self.vec.vocabulary_
=======
>>>>>>> 16dafcaa09f28883e75f23c09ab0bdb064b2d4e0:src/NLP100/util.py
