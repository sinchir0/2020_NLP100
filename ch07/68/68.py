# 68. Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．さらに，クラスタリング結果をデンドログラムとして可視化せよ．

import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

class Data:
    
    def __init__(self):
        pass

    def load_country_vec(self):
        return np.load('../67/country_vec_arr.npy')
    
    def load_countries_list(self):
        return pickle.load(open('../67/countries_list.txt', 'rb'))


if __name__ == "__main__":
    
    # # 国名に関する単語ベクトル
    # country_vec_arr = np.load('../67/country_vec_arr.npy')

    # # 国名のlist
    # with open('../67/countries_list.txt', "rb") as f:
    #     countries_list = pickle.load(f)

    # データの読み込み
    data = Data()
    country_vec_arr = data.load_country_vec()
    countries_list = data.load_countries_list()

    # 階層型clusteringの実施
    cluster = linkage(country_vec_arr, method='ward')

    # 結果を可視化
    fig = plt.figure(figsize=(12, 6))
    dendrogram(cluster, labels=countries_list)
    plt.title('country_dendrogram')
    plt.savefig('country_dendrogram.png', bbox_inches='tight', dpi=300)