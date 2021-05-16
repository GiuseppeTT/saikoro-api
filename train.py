from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import convert_to_d100

SEED = 42
CV = 2

DATA_PATH = Path("data/dice.csv")
MODEL_PATH = Path("model/model.joblib")


def main():
    X_train, X_test, y_train, y_test = load_data()

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = define_model()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    performance = mean_squared_error(model.predict(X_test), y_test, squared=False)

    print(f"Performance: {performance}")


def load_data():
    dice = pd.read_csv(DATA_PATH)
    X, y = dice.drop("d100", axis="columns"), dice["d100"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED)

    return X_train, X_test, y_train, y_test


def define_model():
    steps = [
        ("preprocessor", StandardScaler()),
        ("regressor", None),
    ]
    parameter_grid = {
        "regressor": [
            LinearRegression(),
            RandomForestRegressor(random_state=SEED),
        ],
    }

    pipeline = Pipeline(steps)
    transformed_pipeline = TransformedTargetRegressor(pipeline, inverse_func=convert_to_d100)
    grid_search = GridSearchCV(transformed_pipeline, param_grid=parameter_grid, cv=CV)

    return grid_search


if __name__ == "__main__":
    main()
