# 69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ

import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from adjustText import adjust_text

if __name__ == "__main__":

    # 国名に関する単語ベクトル
    country_vec_arr = np.load('../67/country_vec_arr.npy')

    # 国名のlist
    with open('../67/countries_list.txt', "rb") as f:
        countries_list = pickle.load(f)

    # tsneの実施
    tsne = TSNE(n_components=2, random_state = 33)

    country_embedded = tsne.fit_transform(country_vec_arr)

    # 可視化
    plt.scatter(country_embedded[:,0], country_embedded[:,1])

    texts = [plt.text(
        country_embedded[i,0],
        country_embedded[i,1],
        countries_list[i],
        fontsize=6
        ) for i in range(len(countries_list))]

    adjust_text(texts)    

    plt.savefig('tsne.png', bbox_inches='tight', dpi=300)