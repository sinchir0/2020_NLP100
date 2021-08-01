# 34. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

# 最長一致・・・複数候補ある場合、最も長いものを採用する

from ipdb import set_trace as st

if __name__ == "__main__":

    filename = '../30/neko.txt.mecab'

    sentences = []
    morphs = []
    max_len = 0

    with open(filename, mode='r') as f:
        for line in f: # 1行ずつ読込
            if line != 'EOS\n': # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
                fields = line.split('\t')
                if len(fields) != 2 or fields[0] == '': # ['', '記号,一般,*,*,*,*,*\n']といったように、単語と分かち書きの情報が二つに分かれて入る
                    continue
                else:
                    attr =  fields[1].split(',')
                    if (attr[0] == '名詞') & (attr[6] != '*\n'):
                        morphs.append(attr[6])
                    else:
                        if max_len < len(morphs):
                            max_len = len(morphs)
                            ans = morphs
                        morphs = []
    
    print(ans)
    # ['明治', '三', '十', '八', '年', '何', '月', '何', '日', '戸締り']