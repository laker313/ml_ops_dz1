from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from app.models.models import MODEL_CLASSES, Models
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel
import pandas as pd


router = APIRouter(prefix="/models", tags=["models"])


class TrainRequest(BaseModel):
    hyperparameters: Dict[str, Any] = {}
    # Можно добавить другие параметры:
    # dataset_path: str
    # validation_split: float = 0.2

@router.get("/list")
def get_all_models():
    model_list = [model.value for model in Models]
    return {"message": model_list}





@router.post("/train-with-data")
def train_model(
    model_name: str,
    task_type: str,
    request: TrainRequest
):

    """Создать и обучить модель с указанными гиперпараметрами"""
    try:
        validate_model_name_classifier(model_name, task_type)
        validate_hyperparams(model_name, task_type, request.hyperparameters)
        # Создаем модель с гиперпараметрами
        model = create_model(
            model_name=model_name,
            task_type=task_type,  # или из запроса
            **request.hyperparameters
        )
        
        # Здесь логика обучения модели
        # model.fit(X_train, y_train)

        
        return {
            "status": "success",
            "model_name": request.model_name,
            "hyperparameters": request.hyperparameters,
            "model_type": type(model).__name__
        }
        
    except TypeError or ValueError as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hyperparameters for {request.model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_model_default_params(model_name: str, task_type: str = 'classifier') -> Dict[str, Any]:
    """
    Получить все гиперпараметры по умолчанию для модели
    """
    # Создаем модель с параметрами по умолчанию
    model = create_model(model_name, task_type)
    
    if model is None:
        return {}
    
    # Получаем параметры модели
    if hasattr(model, 'get_params'):
        return model.get_params()
    elif hasattr(model, 'get_all_params'):  # Для catboost
        return model.get_all_params()
    else:
        # Альтернативный способ через __dict__
        return {k: v for k, v in model.__dict__.items() if not k.startswith('_')}

def create_model(model_name: str, task_type: str = 'classifier', **kwargs) -> Any:
    """
    Фабрика для создания моделей
    """

    
    model_class = MODEL_CLASSES[model_name][task_type]
    return model_class(**kwargs)

def validate_model_name_classifier(model_name, task_type):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model {model_name} not supported. Available: {list(MODEL_CLASSES.keys())}")
    
    if task_type not in MODEL_CLASSES[model_name]:
        raise ValueError(f"Task type {task_type} not supported for {model_name}")

def validate_hyperparams(model_name, task_type,params):
    for param in params:
        default_params = get_model_default_params(model_name, task_type)
        if param not in default_params:
            raise ValueError(f"Hyperparameter {param} not valid for {model_name} with task {task_type}")

def get_model_hyperparameters(model_name: str, task_type: str = 'classifier'):
    """Получить все гиперпараметры модели динамически"""
    available_models = [m.value for m in Models]
    if model_name not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_name} not found. Available: {available_models}"
        )
    
    try:
        # Получаем параметры по умолчанию динамически
        params = get_model_default_params(model_name, task_type)
        
        return {
            "model_name": model_name,
            "task_type": task_type,
            "hyperparameters": params
        }
    except Exception as e:
        # print(f"Error getting parameters for {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting parameters for {model_name}: {str(e)}"
        )





