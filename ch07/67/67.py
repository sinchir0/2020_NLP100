# 67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

import pandas as pd
import numpy as np

import gensim
from sklearn.cluster import KMeans

if __name__ == "__main__":
    model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('../64/questions-words.txt') as f:
        questions_words = f.readlines()

    # 重複ありでの国名をquestions-wordsの1,3列目から取得
    countries_dep_list = []

    for line in questions_words:
        if line.startswith(": capital-common-countries"):
            ctg = "capital-common-countries"
        elif line.startswith(":"):
            ctg = "others"
        else:
            if ctg == "capital-common-countries":
                country_1 = line.split(' ')[1]
                country_3 = line.split(' ')[3].replace('\n','')
                countries_dep_list.append(country_1)
                countries_dep_list.append(country_3)
            elif ctg == "others":
                break

    # 重複なしでの国名を取得
    countries_list = list(set(countries_dep_list))

    # 国名のvectorを取得
    vec_list = []
    for country in countries_list:
        country_vec = model[country]
        vec_list.append(country_vec)
    
    # kmeansの実施
    vec_arr = np.array(vec_list)
    kmeans = KMeans(n_clusters=5, random_state=33).fit(vec_arr)

    # 見やすく表示
    print(
        pd.DataFrame(
            {'label':kmeans.labels_,
             'coutry':countries_list})
            .sort_values('label')
            )

    #     label       coutry
    # 0       0         Iran
    # 4       0       Russia
    # 6       0        Egypt
    # 8       0         Cuba
    # 16      1    Australia
    # 13      1      England
    # 21      1       Canada
    # 19      2       Sweden
    # 18      2      Finland
    # 17      2        Italy
    # 14      2        Spain
    # 12      2      Germany
    # 22      2  Switzerland
    # 9       2       Greece
    # 7       2       Norway
    # 5       2       France
    # 15      3        China
    # 1       3        Japan
    # 10      3     Thailand
    # 11      3      Vietnam
    # 3       4  Afghanistan