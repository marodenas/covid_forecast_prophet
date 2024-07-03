from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class Data(BaseModel):
    feature1: float
    feature2: float
    feature3: float

model = joblib.load("path_to_your_model.pkl")

@app.post("/predict")
def predict(data: Data):
    input_data = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}
