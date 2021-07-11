# 12. 1列目をcol1.txtに，2列目をcol2.txtに保存
# 各行の1列目だけを抜き出したものをcol1.txtに，
# 2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．

if __name__ == "__main__":

    with open("../10/popular-names.txt") as f:
        popular_names = f.readlines()

    popular_names_txt1 = [text.split("\t")[0] for text in popular_names]
    popular_names_txt2 = [text.split("\t")[1] for text in popular_names]

    with open("col1.txt", "w") as f:
        f.write("\n".join(popular_names_txt1))

    with open("col2.txt", "w") as f:
        f.write("\n".join(popular_names_txt2))

    # UNIXコマンド
    # cut -f 1 ../10/popular-names.txt
    # cut -f 2 ../10/popular-names.txt
