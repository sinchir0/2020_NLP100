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
    print(sentence.text)
    return translate_api(sentence.text)
