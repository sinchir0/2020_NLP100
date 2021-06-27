# 69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ

import pickle

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.manifold import TSNE


class Data:
    def __init__(self):
        pass

    def load_country_vec(self):
        return np.load("../67/country_vec_arr.npy")

    def load_countries_list(self):
        return pickle.load(open("../67/countries_list.txt", "rb"))


if __name__ == "__main__":

    # データの読み込み
    data = Data()
    country_vec_arr = data.load_country_vec()
    countries_list = data.load_countries_list()

    # tsneの実施
    tsne = TSNE(n_components=2, random_state=33)

    country_embedded = tsne.fit_transform(country_vec_arr)

    # 可視化
    plt.scatter(country_embedded[:, 0], country_embedded[:, 1])

    texts = [
        plt.text(
            country_embedded[i, 0],
            country_embedded[i, 1],
            countries_list[i],
            fontsize=6,
        )
        for i in range(len(countries_list))
    ]

    adjust_text(texts)

    plt.savefig("tsne.png", bbox_inches="tight", dpi=300)
