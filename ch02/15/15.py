# 15. 末尾のN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．

import sys

if __name__ == "__main__":

    N = int(sys.argv[1])

    with open("../13/col1_col2_merge.txt") as f:
        col1_col2_merge = f.readlines()

    print("".join(col1_col2_merge[-N:]))

    # tail ../13/col1_col2_merge.txt