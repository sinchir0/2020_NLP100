# 11. タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

if __name__ == "__main__":

    with open("../10/popular-names.txt") as f:
        popular_names = f.readlines()

    popular_names = [text.replace("\t", " ") for text in popular_names]

    print("".join(popular_names))

    # UNIXコマンド
    # cat  ../10/popular-names.txt | tr '\t' ' '
