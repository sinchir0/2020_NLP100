# 言語処理１００本ノック　2020をやっていきます。
https://nlp100.github.io/ja/ch01.html

下記の方々の回答を非常に参考にさせて頂いております。
<br>
wakameさん
https://github.com/wakamezake/nlp_q100_2020
<br>
u++さん
https://github.com/upura/nlp100v2020
<br>
taro-masudaさん
https://github.com/taro-masuda/nlp100
<br>
takapy0210さん
https://github.com/takapy0210/nlp_2020

# docker環境の立ち上げ方

```
# buildとrun
# -dはbackgroundでの起動
docker-compose up --build -d 

# runだけしたい場合
docker-compose up -d

# docker-composeで構築した環境の確認方法
docker-compose ps

# fishでのdocker containerへの接続
# appの部分はdocker-compose.ymlで指定したservice名。存在しない名前だと、何も起きないため注意。
docker-compose exec app fish 

# network, container 削除
docker-compose down

# network, container, images 削除
docker-compose down --rmi all
```
