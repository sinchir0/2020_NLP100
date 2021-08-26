# サーバーの立ち上げ方
uvicorn api:app --reload

# 翻訳する文章の投げ方
curl -X POST -H "Content-Type: application/json" -d '{"text":"私はリンゴが好きです"}' localhost:8000/predict