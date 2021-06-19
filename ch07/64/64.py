# 64. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．

# !wget http://download.tensorflow.org/data/questions-words.txt

import gensim

if __name__ == "__main__":
    model = gensim.models.KeyedVectors.load_word2vec_format('../60/GoogleNews-vectors-negative300.bin', binary=True)

    with open('questions-words.txt') as f:
        questions_words = f.readlines()

    result_list = []

    for row in range(len(questions_words)):
        org_line = questions_words[row]

        first_word = org_line.split(' ')[0]
        if first_word == ':':
            result_list.append(org_line)
            continue
        second_word = org_line.split(' ')[1]
        third_word = org_line.split(' ')[2]

        result = model.most_similar(positive=[second_word, third_word], negative=[first_word], topn=1)

        org_line_add_result = org_line[:-1] + ' ' + str(result[0])
        
        result_list.append(org_line_add_result)

    join_result = '\n'.join(result_list)
    
    with open("result.txt", 'w') as f:
        f.write(join_result)