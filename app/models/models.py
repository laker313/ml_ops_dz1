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



class Model_Type(Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"



MODEL_CLASSES = {
    Models.CATBOOST.value: {
        Model_Type.CLASSIFIER.value: CatBoostClassifier,
        Model_Type.REGRESSOR.value : CatBoostRegressor
    },
    Models.LIGHTGBM.value: {
        Model_Type.CLASSIFIER.value : LGBMClassifier, 
        Model_Type.REGRESSOR.value : LGBMRegressor
    },
    Models.XGBOOST.value: {
        Model_Type.CLASSIFIER.value: XGBClassifier,
        Model_Type.REGRESSOR.value : XGBRegressor
    },
    Models.RANDOM_FOREST.value: {
        Model_Type.CLASSIFIER.value : RandomForestClassifier,
        Model_Type.REGRESSOR.value : RandomForestRegressor
    },
    Models.LINEAR_REGRESSION.value: {
        Model_Type.REGRESSOR.value : LinearRegression,
        Model_Type.CLASSIFIER.value : LogisticRegression
    }
}