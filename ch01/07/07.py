# 07. テンプレートによる文生成
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def make_sentece_from_template(x, y, z):
    return f'{x}時の{y}は{z}'

if __name__ == "__main__":
    print(make_sentece_from_template(12, '気温', 22.4))