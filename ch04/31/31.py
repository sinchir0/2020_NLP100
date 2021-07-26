# 31. 動詞
# 動詞の表層形をすべて抽出せよ．

from ipdb import set_trace as st

if __name__ == "__main__":

    filename = '../30/neko.txt.mecab'

    sentences = []
    morphs = []

    with open(filename, mode='r') as f:
        for line in f:  # 1行ずつ読込
            if line != 'EOS\n':  # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
                fields = line.split('\t')
                if len(fields) != 2 or fields[0] == '': # ['', '記号,一般,*,*,*,*,*\n']といったように、単語と分かち書きの情報が二つに分かれて入る
                    continue
                else:
                    attr =  fields[1].split(',')
                    if attr[0] == '動詞':
                        print(fields[0]) # 動詞の表層形を表示
                        # 見る
                        # なる
                        # 見える
                        # し
                        # くる
                        # い
                        # 来る
                        # 来
                        # 云う
                        # 得よ
                        # 思い
                        # 詣り