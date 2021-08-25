# 36. 頻度上位10語
# 出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import pickle
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

if __name__ == "__main__":

    with open('../35/neko_list', mode="rb") as f:
        neko_list = pickle.load(f)

    word_list = []
    word_list = [morphs[0] for morphs in neko_list]
    
    c = collections.Counter(word_list)

    c_top10 = c.most_common(10)
    word_top10 = [value[0] for value in c_top10]
    freq_top10 = [value[1] for value in c_top10]

    print(c_top10)
    # [('の', 9194), ('。', 7486), ('て', 6868), ('、', 6772), ('は', 6420), ('に', 6243), ('を', 6071), ('と', 5508), ('が', 5337), ('た', 3988)]
    
    # グラフ描画
    plt.bar(word_top10, freq_top10)
    plt.savefig('counter.png')
    plt.close()