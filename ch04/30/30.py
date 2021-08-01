# 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
# ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
# 1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

# mecabでの分かち書き
# mecab -o ./neko.txt.mecab ./neko.txt

if __name__ == "__main__":

    filename = "./neko.txt.mecab"

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
                    morph = {
                        "surface": fields[0],
                        "base": attr[6],
                        "pos": attr[0],
                        "pos1": attr[1],
                    }
                    morphs.append(morph)
            else:  # 文末：形態素リストを文リストに追加
                sentences.append(morphs)
                morphs = []

    # 確認
    for morph in sentences[2]:
        print(morph)
