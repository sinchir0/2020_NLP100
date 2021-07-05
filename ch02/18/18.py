# 18. 各行を3コラム目の数値の降順にソート
# 各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．


def file_line_reader_generator(file_path):
    """ファイルの行を返すジェネレータ"""
    with open(file_path, encoding="utf-8") as in_file:
        for line in in_file:
            yield line


if __name__ == "__main__":
    # generatorで読み込む
    popular_names = file_line_reader_generator("../10/popular-names.txt")

    popular_names = sorted(popular_names, key=lambda x: x.split("\t")[2], reverse=True)

    for name in popular_names:
        print(name)

    # UNIXコマンド
    # sort -n -k 3 -r ../10/popular-names.txt
    # -n: 文字列を数値と見なして並べ替える
    # -k 指定:　場所と並び替え種別を指定する
    # -r: 逆順に並び替える
