import MeCab
from fastapi import FastAPI
from pydantic import BaseModel

from translate import translate_api

app = FastAPI()


class Sentence(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def translate(sentence: Sentence):
    mecab = MeCab.Tagger("-Owakati")
    parse_text = mecab.parse(sentence.text)
    return translate_api(parse_text)
