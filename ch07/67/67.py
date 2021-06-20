# 67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

import pickle

import pandas as pd
import numpy as np

import gensim
from sklearn.cluster import KMeans

def get_country_name(questions_words: list, category_name: str) -> list:
    """重複ありでの国名をquestions-wordsの: capital-common-countries, 1,3列目から取得

    Args:
        questions_words (list): questions_words.txtのリスト
        category_name (str): questions_words.txtの：で区切られるジャンル
    return:
        list: 国名のリスト
    """

    countries_set = set()

    for line in questions_words:
        if line.startswith(f": {category_name}"):
            ctg = category_name
        elif line.startswith(":"):
            ctg = "others"
        else:
            if ctg == category_name:
                country_1 = line.split(' ')[1]
                country_3 = line.split(' ')[3].replace('\n','')
                countries_set.add(country_1)
                countries_set.add(country_3)
            elif ctg == "others":
                continue

    # 国名のlist
    countries_list = list(countries_set)

    return countries_list

if __name__ == "__main__":
    model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('../64/questions-words.txt') as f:
        questions_words = f.readlines()

    # 「capital-common-countries」「capital-world」の区切りから国名を取得
    common_countries_list = get_country_name(questions_words, "capital-common-countries")
    world_countries_list = get_country_name(questions_words, "capital-world")

    # 重複を無くした国名を一つのlistにまとめる
    countries_list = list(set(common_countries_list + world_countries_list))

    # 国名のvectorを取得
    vec_list = [model[country] for country in countries_list]
    
    # kmeansの実施
    country_vec_arr = np.array(vec_list)
    kmeans = KMeans(n_clusters=5, random_state=33).fit(country_vec_arr)

    # 保存
    np.save('country_vec_arr', country_vec_arr)

    with open('countries_list.txt', "wb") as f:
        pickle.dump(countries_list, f)

    # 見やすく表示
    print(
        pd.DataFrame(
            {'label':kmeans.labels_,
             'coutry':countries_list})
            .sort_values('label')
            )

#     label       coutry
# 87      0      Algeria
# 24      0   Mozambique
# 25      0       Malawi
# 28      0         Mali
# 85      0     Botswana
# ..    ...          ...
# 22      4      Austria
# 91      4  Switzerland
# 92      4         Iraq
# 76      4       Jordan
# 86      4      Morocco