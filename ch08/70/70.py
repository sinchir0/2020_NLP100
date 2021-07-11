# 70. 単語ベクトルの和による特徴量

import gensim
import numpy as np
from NLP100.util import load_data, preprocess


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
