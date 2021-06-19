# 66. WordSimilarity-353での評価
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし，単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．

# !wget http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
# !unzip wordsim353.zip -d wordsim353

import gensim
from scipy import stats

if __name__ == "__main__":
    model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('wordsim353/combined.csv') as f:
        next(f) # headerは読み込まない
        combined = f.readlines()
    
    human_sim_list = []
    wordvec_sim_list = []

    for line in combined:
        # 単語ベクトルにより計算される類似度のリストの作成
        first_word = line.split(',')[0]
        second_word = line.split(',')[1]
        wordvec_sim = model.similarity(first_word, second_word)
        wordvec_sim_list.append(wordvec_sim)

        # 人間の類似度判定のリストの作成
        human_sim = float(line.split(',')[2][:-2])
        human_sim_list.append(human_sim)

    print(stats.spearmanr(wordvec_sim_list, human_sim_list))