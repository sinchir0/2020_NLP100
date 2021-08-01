# 33. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．

# 二つ前の文字の品詞=「名詞」& 一つ前の文字の表層形=「の」& 一つ前の文字の品詞=「助詞」& 現在の文字の品詞=「名詞」

from ipdb import set_trace as st

if __name__ == "__main__":

    filename = "../30/neko.txt.mecab"

    sentences = []
    morphs = []

    with open(filename, mode="r") as f:
        all_text = [line for line in f]
        # for line in f:  # 1行ずつ読込
        for line1, line2, line3 in zip(all_text, all_text[1:], all_text[2:]):  # 1行ずつ読込
            if line2 != "EOS\n":  # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
                fields = line2.split("\t")
                if (
                    len(fields) != 2 or fields[0] == ""
                ):  # ['', '記号,一般,*,*,*,*,*\n']といったように、単語と分かち書きの情報が二つに分かれて入る
                    continue
                else:
                    attr = fields[1].split(",")
                    if (fields[0] == "の") & (attr[0] == "助詞"):

                        fields1 = line1.split("\t")
                        fields3 = line3.split("\t")
                        if (len(fields1) != 2 or fields1[0] == "") | (
                            len(fields3) != 2 or fields3[0] == ""
                        ):
                            continue
                        else:
                            st()
                            attr1 = fields1[1].split(",")
                            attr3 = fields3[1].split(",")
                            if (attr1[0] == "名詞") & (attr3[0] == "名詞"):
                                print(line1, line2, line3)
                            # 元信    名詞,固有名詞,人名,名,*,*,元信,モトノブ,モトノブ
                            # の     助詞,連体化,*,*,*,*,の,ノ,ノ
                            # 幅     名詞,一般,*,*,*,*,幅,ハバ,ハバ

                            # 客      名詞,一般,*,*,*,*,客,キャク,キャク
                            # の     助詞,連体化,*,*,*,*,の,ノ,ノ
                            # 間     名詞,一般,*,*,*,*,間,マ,マ

                            # 三      名詞,数,*,*,*,*,三,サン,サン
                            # の     助詞,連体化,*,*,*,*,の,ノ,ノ
                            # 問答   名詞,サ変接続,*,*,*,*,問答,モンドウ,モンドー
