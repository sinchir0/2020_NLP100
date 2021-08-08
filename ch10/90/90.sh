#!/bin/sh

# 90. データの準備
# 機械翻訳のデータセットをダウンロードせよ．
# 訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．
# ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．

# https://qiita.com/nymwa/items/867e05a43060d036a174

# wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
# tar -zxvf kftt-data-1.0.tar.gz

# cat kftt-data-1.0/data/orig/kyoto-train.ja | sed 's/\s+/ /g' | ginzame > train.ginza.ja
# cat kftt-data-1.0/data/orig/kyoto-dev.ja | sed 's/\s+/ /g' | ginzame > dev.ginza.ja
# cat kftt-data-1.0/data/orig/kyoto-test.ja | sed 's/\s+/ /g' | ginzame > test.ginza.ja

# sed 's/\s+/ /g'のsedは置換コマンド　s/検索パターン/置換文字列/g
# \sは空白文字、+は1文字以上の繰り返し
# つまり、空白の長さを１文字に揃える。

mecab -Owakati kftt-data-1.0/data/orig/kyoto-train.ja -o train.mecab.ja
mecab -Owakati kftt-data-1.0/data/orig/kyoto-dev.ja -o dev.mecab.ja
mecab -Owakati kftt-data-1.0/data/orig/kyoto-test.ja -o test.mecab.ja