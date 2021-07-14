# 13. col1.txtとcol2.txtをマージ
# 12で作ったcol1.txtとcol2.txtを結合し，
# 元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
# 確認にはpasteコマンドを用いよ．

if __name__ == "__main__":

    with open("../12/col1.txt") as f:
        col1 = f.readlines()

    with open("../12/col2.txt") as f:
        col2 = f.readlines()

    col1_col2_merge = [
        col1.replace("\n", "") + "\t" + col2.replace("\n", "")
        for col1, col2 in zip(col1, col2)
    ]

    with open("col1_col2_merge.txt", "w") as f:
        f.write("\n".join(col1_col2_merge))

    # UNIXコマンド
    # paste col1_col2_merge.txt
