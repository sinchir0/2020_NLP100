# 19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
# 各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．
# 確認にはcut, uniq, sortコマンドを用いよ．

from collections import Counter

def file_line_reader_generator(file_path):
    """ファイルの行を返すジェネレータ"""
    with open(file_path, encoding="utf-8") as in_file:
        for line in in_file:
            yield line

if __name__ == "__main__":

    # generatorで読み込む
    popular_names = file_line_reader_generator("../10/popular-names.txt")

    #for name in popular_names:
    #    col1 = name.split('\t')[3]
    #    col1_set.add(col1)

    name_list = [name.split('\t')[0] for name in popular_names]
    print(Counter(name_list))

    # UNIXの例
    # cut -f 1 ../10/popular-names.txt | sort | uniq -c | sort -r
    # uniq -c: 重複している行を数える
    # sort -r: 逆順に並べ替える