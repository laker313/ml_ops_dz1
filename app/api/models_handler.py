from fastapi import FastAPI, File, Response, UploadFile, Form, HTTPException, APIRouter
from scipy import io
from app.models.models import MODEL_CLASSES, Models
from app.data_storage.mino.model_storage import delete_model_from_minio, save_model_to_minio, read_model_from_minio, update_model_to_minio
from app.data_storage.mino.datasets_storage import read_dataset_from_minio
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel
import pandas as pd

from ml_ops_dz1.app.api.dataset_handler import dataset_validation, format_validation, read_pd_from_format
from ml_ops_dz1.app.data_storage.model.model_status import Learning_status


router = APIRouter(prefix="/models", tags=["models"])


class ModelRequest(BaseModel):
    hyperparameters: Dict[str, Any] = {}
    # Можно добавить другие параметры:
    # dataset_path: str
    # validation_split: float = 0.2

@router.get("/type_list")
async def get_all_models():
    model_list = [model.value for model in Models]
    return {"message": model_list}



@router.post("/create_and_save_model")
async def create_and_save_model(
    model_name: str,
    task_type: str,
    request: ModelRequest
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
        
        model_id = save_model_to_minio(model, model_name, request.hyperparameters)

        
        return {
            "status": "saved",
            "model_name": request.model_name,
            "hyperparameters": request.hyperparameters,
            "model_type": type(model).__name__,
            "model_id": model_id
        }
        
    except TypeError or ValueError as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hyperparameters for {request.model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/learn_model")
async def learn_model(
    model_id: str,
    data_id: str
):
    try:
        model = read_model_from_minio(model_id)

        dataset = pd.read_parquet(io.BytesIO(read_dataset_from_minio(data_id)))
        # Здесь логика обучения модели
        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        model.fit(X, y)

        update_model_to_minio(model, model_id, model_id.model_name, {}, Learning_status.LEARNED)
        return {
            "status": f"model_{model_id}_learned_with_data_{data_id}",
            "model_type": type(model).__name__,
            "model_id": model_id
        }
        
    except TypeError or ValueError as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data for {model_id.model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/update_model")
async def create_and_save_model(
    model_name: str,
    task_type: str,
    request: ModelRequest
):

    """Создать и обучить модель с указанными гиперпараметрами"""
    try:
        validate_model_name_classifier(model_name, task_type)
        validate_hyperparams(model_name, task_type, request.hyperparameters)
        # Создаем модель с гиперпараметрами
        model = create_model(
            model_name=model_name,
            task_type=task_type,
            **request.hyperparameters
        )
        
        model_id = update_model_to_minio(model, model_name, request.hyperparameters, Learning_status.NOT_LEARNED)

        
        return {
            "status": "model_updated",
            "model_name": request.model_name,
            "hyperparameters": request.hyperparameters,
            "model_type": type(model).__name__,
            "model_id": model_id
        }
        
    except TypeError or ValueError as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hyperparameters for {request.model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@router.post("/delete_model")
async def delete_model(
    model_id: str
):

    """Создать и обучить модель с указанными гиперпараметрами"""
    try:
        deleted = delete_model_from_minio(model_id)

        
        return {
            "status": f"model_deleted",
            "model_id": model_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_predictions_from_file")
async def learn_model(
    model_id: str,
    file: UploadFile = File(...)
):
    try:
        resp = read_model_from_minio(model_id)
        learning_status = resp["learning_status"]
        model = resp["model"]
        if learning_status != Learning_status.LEARNED:
            raise ValueError(f"Model {model_id} is not learned yet.")
        file_format = format_validation(file)
        try:
            # Читаем файл
            dataset = await read_pd_from_format(file, file_format)
            dataset["target"] = 0  # Заглушка для совместимости
            dataset_X = dataset.drop(columns=["target"])
            predictions = model.predict(dataset_X)
            # dataset_validation(dataset) наверное нужна но пока не знаю какая
            dataset_bytes = io.BytesIO()
            pd.DataFrame(predictions, columns=["predictions"]).to_parquet(dataset_bytes, engine='pyarrow', index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading input file: {str(e)}")
        return Response(
            content=dataset_bytes,
            media_type="application/parquet",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename}.parquet"
            }
        )
        
    except TypeError or ValueError as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data for {model_id.model_name}: {str(e)}"
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





