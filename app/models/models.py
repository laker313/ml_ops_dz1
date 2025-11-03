from enum import Enum


from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from enum import Enum

class Models(Enum):
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"




MODEL_CLASSES = {
    Models.CATBOOST.value: {
        'classifier': CatBoostClassifier,
        'regressor': CatBoostRegressor
    },
    Models.LIGHTGBM.value: {
        'classifier': LGBMClassifier, 
        'regressor': LGBMRegressor
    },
    Models.XGBOOST.value: {
        'classifier': XGBClassifier,
        'regressor': XGBRegressor
    },
    Models.RANDOM_FOREST.value: {
        'classifier': RandomForestClassifier,
        'regressor': RandomForestRegressor
    },
    Models.LINEAR_REGRESSION.value: {
        'regressor': LinearRegression,
        'classifier': LogisticRegression
    }
}