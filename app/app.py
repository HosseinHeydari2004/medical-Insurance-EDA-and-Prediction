from pydantic import BaseModel
from fastapi import FastAPI
import joblib
from pandas import DataFrame, get_dummies
from numpy import array

app = FastAPI()


class Insurance_expenses(BaseModel):
    age: int
    gender: str
    bmi: float
    children: int
    discount_eligibility: str
    region: str
    premium: float
    
class Insurance_premium(BaseModel):
    age: int
    gender: str
    bmi: float
    children: int
    discount_eligibility: str
    region: str
    expenses: float

model_expenses = joblib.load(r"B:\AI Source\Projects\list_of_projects\پروژه های تمرینی\medical-Insurance-EDA-and-Prediction\models\pipeline_expenses.joblib")
model_premium = joblib.load(r"B:\AI Source\Projects\list_of_projects\پروژه های تمرینی\medical-Insurance-EDA-and-Prediction\models\pipeline_premium.joblib")


EXPECTED_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "children",
    "discount_eligibility",
    "premium",
    "region_northeast",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]

@app.post("/predict_expenses")
def predict_expenses(data: Insurance_expenses):
    dt = DataFrame(data=[data.model_dump()])
    df = get_dummies(
    data=dt, columns=["region"]
    )
    df.replace(
    {"gender": {"male": 1, "female": 0}, "discount_eligibility": {"no": 0, "yes": 1}},
    inplace=True,
    )
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    prediction = model_expenses.predict(df)
    return {
        "prediction": prediction[0]
    }

@app.post("/predict_premium")
def predict_premium(data: Insurance_premium):
    dt = DataFrame(data=data)
    df = get_dummies(
    data=dt, columns=["region"]
    )
    df.replace(
    {"gender": {"male": 1, "female": 0}, "discount_eligibility": {"no": 0, "yes": 1}},
    inplace=True,
    )
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    prediction = model_premium.predict(df)
    
    return {
        "prediction": prediction[0]
    }