# 10. 行数のカウントPermalink
# 行数をカウントせよ．確認にはwcコマンドを用いよ．

# !wget https://nlp100.github.io/data/popular-names.txt

if __name__ == "__main__":

    with open("popular-names.txt") as f:
        popular_names = f.readlines()

    print(len(popular_names))
    # UNIXコマンド
    # wc popular-names.txt