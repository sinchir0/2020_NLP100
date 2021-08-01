# 70. 単語ベクトルの和による特徴量

import gensim
import numpy as np
import pandas as pd
import texthero as hero


def load_data(data_check=False) -> dict:
    """データの読み込み"""
    # 読み込むファイルを定義
    inputs = {
        "train": "../../ch06/50/train.txt",
        "valid": "../../ch06/50/valid.txt",
        "test": "../../ch06/50/test.txt",
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
    return clean_text


def get_mean_vector(model, sentence: list):
    # remove out-of-vocabulary words
    words = [word for word in sentence if word in model.vocab]
    if len(words) >= 1:
        return np.mean(model[words], axis=0)
    else:
        return []


if __name__ == "__main__":

    name_list = ["train", "valid", "test"]

    dfs = load_data()

    model = gensim.models.KeyedVectors.load_word2vec_format(
        "../../ch07/60/GoogleNews-vectors-negative300.bin", binary=True
    )

    # 前処理
    for name in name_list:
        dfs[name]["TITLE"] = dfs[name][["TITLE"]].apply(preprocess)

    # 分かち書き
    for name in name_list:
        dfs[name]["TITLE_SPLIT"] = [
            text.split(" ") for text in dfs[name]["TITLE"].tolist()
        ]

    # 特徴量行列の取得
    for name in name_list:
        dfs[name]["TITLE_VECTOR"] = [
            get_mean_vector(model, text) for text in dfs[name]["TITLE_SPLIT"].tolist()
        ]

    # ラベル変換
    label_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
    for name in name_list:
        dfs[name]["CATEGORY"] = dfs[name]["CATEGORY"].map(label_dict)

    # データの保存
    for name in name_list:
        # 特徴量行列
        np.save(f"{name}_vector", np.stack(dfs[name]["TITLE_VECTOR"]))
        # ラベル
        np.save(f"{name}_label", np.stack(dfs[name]["CATEGORY"]))
