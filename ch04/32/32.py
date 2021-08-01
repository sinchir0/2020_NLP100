# 32. 動詞の基本形
# 動詞の基本形をすべて抽出せよ．

from ipdb import set_trace as st

if __name__ == "__main__":

    filename = "../30/neko.txt.mecab"

    sentences = []
    morphs = []

    with open(filename, mode="r") as f:
        for line in f:  # 1行ずつ読込
            if line != "EOS\n":  # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
                fields = line.split("\t")
                if (
                    len(fields) != 2 or fields[0] == ""
                ):  # ['', '記号,一般,*,*,*,*,*\n']といったように、単語と分かち書きの情報が二つに分かれて入る
                    continue
                else:
                    attr = fields[1].split(",")
                    if attr[0] == "動詞":
                        print(attr[6])  # 動詞を表示
                        # 逃がす
                        # 続く
                        # 飛び下りる
                        # 見える
                        # 捕る
                        # 思う
                        # 捕る
                        # 知る
                        # 廻る
                        # 出す
                        # する
                        # 流す
