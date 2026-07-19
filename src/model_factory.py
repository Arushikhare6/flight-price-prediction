from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


RANDOM_STATE = 42


def build_models():

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),

        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }

    return models