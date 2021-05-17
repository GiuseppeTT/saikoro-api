from pathlib import Path
from typing import Optional, Type

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConstrainedInt
from utils import convert_to_d100  # Necessary for joblib load model

MODEL_PATH = Path("model/model.joblib")


class D4(ConstrainedInt):
    ge = 1
    le = 4


class D6(ConstrainedInt):
    ge = 1
    le = 6


class D8(ConstrainedInt):
    ge = 1
    le = 8


class D10(ConstrainedInt):
    ge = 1
    le = 10


class D12(ConstrainedInt):
    ge = 1
    le = 12


class D20(ConstrainedInt):
    ge = 1
    le = 20


class D100(ConstrainedInt):
    ge = 10
    le = 100
    multiple_of = 10


class DiceData(BaseModel):
    d4: list[Optional[D4]]
    d6: list[Optional[D6]]
    d8: list[Optional[D8]]
    d10: list[Optional[D10]]
    d12: list[Optional[D12]]
    d20: list[Optional[D20]]


model = joblib.load(MODEL_PATH)

app = FastAPI()


@app.post("/predict")
def predict(data: DiceData) -> list[D100]:
    X = pd.DataFrame(dict(data))
    prediction = model.predict(X).tolist()

    return prediction
