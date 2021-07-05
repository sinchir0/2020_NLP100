# 14. 先頭からN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ

import sys

if __name__ == "__main__":

    N = int(sys.argv[1])

    with open("../13/col1_col2_merge.txt") as f:
        col1_col2_merge = f.readlines()

    print("".join(col1_col2_merge[:N]))

    # head ../13/col1_col2_merge.txt
    # generator
