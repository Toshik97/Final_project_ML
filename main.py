import dill
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

# Проверить работу сервиса можно с использованием json файлов.

app = FastAPI()
with open('model/Model_downsampling.pkl', 'rb') as f:
    model = dill.load(f)


class Form(BaseModel):
    utm_medium: str
    utm_source: str
    utm_campaign: str
    device_category: str
    device_brand: str
    device_browser: str
    geo_country: str
    device_os: str
    geo_city: str
    utm_keyword: str


class Prediction(BaseModel):
    pred: bool


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([form.dict()])
    y = model['model'].predict(df)

    return {
        'pred': y

    }
