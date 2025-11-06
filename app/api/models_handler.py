
from app.models.models import MODEL_CLASSES, Models
from app.data_storage.mino.model_storage import delete_model_from_minio, save_model_to_minio, read_model_from_minio, update_model_to_minio
from app.data_storage.mino.datasets_storage import read_dataset_from_minio
from app.api.dataset_handler import format_validation, read_pd_from_format
from app.data_storage.model.model_status import Learning_status
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel
import pandas as pd
import io
import json
from fastapi import FastAPI, File, Response, UploadFile, Form, HTTPException, APIRouter
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps

# Глобальный пул с разумным лимитом
GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=40, thread_name_prefix="model_worker")

router = APIRouter(prefix="/models", tags=["models"])




def async_run_in_pool(func):
    """Декоратор для запуска синхронных функций в пуле потоков"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(GLOBAL_THREAD_POOL, lambda: func(*args, **kwargs))
    return wrapper


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "workers": len(GLOBAL_THREAD_POOL._threads),
        "queue_size": GLOBAL_THREAD_POOL._work_queue.qsize()
    }



@router.get("/pool_status")
async def pool_status():
    return {
        "max_workers": GLOBAL_THREAD_POOL._max_workers,
        "active": len(GLOBAL_THREAD_POOL._threads),
        "queue": GLOBAL_THREAD_POOL._work_queue.qsize(),
    }


@router.get("/type_list")
async def get_all_models():
    model_list = [model.value for model in Models]
    return {"message": model_list}


@router.post("/create_and_save_model")
@async_run_in_pool
def create_and_save_model(
    model_name: str = Form(...),
    task_type: str = Form(...),
    hyperparameters: str = Form("{}")
):

    """Создать и обучить модель с указанными гиперпараметрами"""
    try:
        params = json.loads(hyperparameters)
        validate_model_name_classifier(model_name, task_type)
        validate_hyperparams(model_name, task_type, params)
        # Создаем модель с гиперпараметрами
        model = create_model(
            model_name=model_name,
            task_type=task_type,  # или из запроса
            **params
        )
        
        model_id = save_model_to_minio(model, model_name, params)

        
        return {
            "status": "saved",
            "model_name": model_name,
            "hyperparameters": params,
            "model_type": type(model).__name__,
            "model_id": model_id
        }
        
    except (TypeError, ValueError) as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hyperparameters for {model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/learn_model")
@async_run_in_pool
def learn_model(
    model_id: str = Form(...),
    data_id: str = Form(...)
):
    try:
        resp = read_model_from_minio(model_id)
        model = resp["model"]
        hyperparams = resp["hyperparams"]
        model_name = resp["model_name"]
        dataset = pd.read_parquet(io.BytesIO(read_dataset_from_minio(data_id)))
        # Здесь логика обучения модели
        if "target" not in dataset.columns:
            raise HTTPException(status_code=400, detail="Dataset must contain a 'target' column for training.")

        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        model.fit(X, y)

        update_model_to_minio(model, model_id, model_name, hyperparams, Learning_status.LEARNED)
        return {
            "status": f"model_{model_id}_learned_with_data_{data_id}",
            "model_type": model_name,
            "model_id": model_id
        }
        
    except (TypeError, ValueError) as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data for {type(model).__name__}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update_model")
@async_run_in_pool
def update_model(
    model_name: str = Form(...),
    task_type: str = Form(...),
    hyperparameters: str = Form("{}")
):

    """Создать и обучить модель с указанными гиперпараметрами"""
    try:
        params = json.loads(hyperparameters)
        validate_model_name_classifier(model_name, task_type)
        validate_hyperparams(model_name, task_type, params)
        # Создаем модель с гиперпараметрами
        model = create_model(
            model_name=model_name,
            task_type=task_type,
            **params
        )
        
        model_id = update_model_to_minio(model, model_name, params, Learning_status.NOT_LEARNED)

        
        return {
            "status": "model_updated",
            "model_name": model_name,
            "hyperparameters": params,
            "model_type": type(model).__name__,
            "model_id": model_id
        }
        
    except (TypeError, ValueError) as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hyperparameters for {model_name}: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.post("/delete_model")
@async_run_in_pool
def delete_model(
    model_id: str = Form(...)
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
@async_run_in_pool
def get_predictions_from_file(
    model_id: str = Form(...),
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
            dataset = read_pd_from_format(file, file_format)
            # dataset["target"] = 0  # Заглушка для совместимости
            if "target" in dataset.columns:
                dataset_X = dataset.drop(columns=["target"])
            else:
                dataset_X = dataset
            predictions = model.predict(dataset_X)
            # dataset_validation(dataset) наверное нужна но пока не знаю какая
            dataset_bytes = io.BytesIO()
            pd.DataFrame(predictions, columns=["predictions"]).to_parquet(dataset_bytes, engine='pyarrow', index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading input file: {str(e)}")
        return Response(
            content=dataset_bytes.getvalue(),
            media_type="application/parquet",
            headers={
                "Content-Disposition": f"attachment; filename=ans.parquet"
            }
        )
        
    except (TypeError, ValueError) as e:
        # Ошибка в гиперпараметрах
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data for {type(model).__name__}: {str(e)}"
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
    default_params = get_model_default_params(model_name, task_type)
    for param in params:
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





