from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter, Response
from app.models.models import MODEL_CLASSES, Models
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from app.data_storage.mino.model_storage import save_model_to_minio
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel
import pandas as pd
from app.data_storage.mino.datasets_storage import save_dataset_to_minio, update_dataset_to_minio
import io
from app.data_storage.mino.datasets_storage import delete_dataset_from_minio, read_dataset_from_minio
from app.api.models_handler import async_run_in_pool

router = APIRouter(prefix="/data", tags=["data"])




EXTENSION_MAP = {
    'csv': 'csv',
    'parquet': 'parquet', 
    'json': 'json',
    'pkl': 'pickle',
    'pickle': 'pickle',
    'feather': 'feather'
}


SUPPORTED_FORMATS = {
    'csv': {'reader': pd.read_csv, 'content_types': ['text/csv', 'application/csv']},
    'parquet': {'reader': pd.read_parquet, 'content_types': ['application/parquet']},
    'json': {'reader': pd.read_json, 'content_types': ['application/json']},
    'pickle': {'reader': pd.read_pickle, 'content_types': ['application/octet-stream']},
    'feather': {'reader': pd.read_feather, 'content_types': ['application/octet-stream']},
}






@router.post("/upload_dataset")
@async_run_in_pool
def upload_dataset(
    file: UploadFile = File(...)
):
    """Загрузить набор данных в хранилище"""
    # Валидация файла
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Определяем формат
    file_format = determine_file_format(file.filename)
    if file_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {list(SUPPORTED_FORMATS.keys())}")
    try:
        # Читаем файл
        dataset = read_pd_from_format(file, file_format)
        dataset_validation(dataset)
        
        dataset_id = save_dataset_to_minio(dataset, file.filename)
        
        return {
            "status": "dataset_uploaded", 
            "dataset_id": dataset_id,
            "dataset_name": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.post("/update_dataset")
@async_run_in_pool
def update_dataset(
    file: UploadFile = File(...),
    dataset_id: str = Form(...)
):
    file_format = format_validation(file)
    try:
        # Читаем файл
        dataset = read_pd_from_format(file, file_format)
        dataset_validation(dataset)
        
        dataset_id = update_dataset_to_minio(dataset, dataset_id, file.filename)
        
        return {
            "status": "dataset_uploaded", 
            "dataset_id": dataset_id,
            "dataset_name": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










def read_pd_from_format(file, file_format) -> pd.DataFrame:
    content = file.read()
    file_bytes = io.BytesIO(content)
    dataset = SUPPORTED_FORMATS[file_format]['reader'](file_bytes)
    return dataset

def format_validation(file):
    """Загрузить набор данных в хранилище"""
    # Валидация файла
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Определяем формат
    file_format = determine_file_format(file.filename)
    if file_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {list(SUPPORTED_FORMATS.keys())}")
            
    return file_format
    
@router.post("/download_dataset")
@async_run_in_pool
def download_dataset(
    dataset_id: str = Form(...)
):
    """Загрузить набор данных в хранилище"""
    try:
        # Читаем файл
        dataset_bytes = read_dataset_from_minio(dataset_id)
        
        
        return Response(
            content=dataset_bytes,
            media_type="application/parquet",
            headers={
                "Content-Disposition": f"attachment; filename={dataset_id}.parquet"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete_dataset")
@async_run_in_pool
def delete_dataset(
    dataset_id: str = Form(...)
):

    try:
        
        delete_dataset_from_minio(dataset_id)
        
        return {
            "status": "dataset_deleted", 
            "dataset_id": dataset_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


def determine_file_format(filename: str) -> str:
    """Определяем формат файла"""
    # Приоритет по расширению файла
    ext = filename.lower().split('.')[-1]
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]
    return "unknown"



def dataset_validation(dataset: pd.DataFrame):
    """
    Проверяет что:
    1. Есть столбец 'target'
    2. Он только один
    3. (опционально) Проверяет значения
    """
    
    # 1. Проверяем наличие столбца 'target'
    if 'target' not in dataset.columns:
        raise ValueError("Dataset must contain 'target' column")
    
    # 2. Проверяем что столбец только один
    target_columns = [col for col in dataset.columns if col == 'target']
    if len(target_columns) > 1:
        raise ValueError("There should be only one 'target' column in the dataset")
    
    # 3. Дополнительные проверки (опционально)
    target_col = dataset['target']
    
    # Проверяем что столбец не пустой
    if target_col.empty:
        raise ValueError("Target column is empty")
    
    # Проверяем что нет пропущенных значений
    if target_col.isnull().any():
        raise ValueError("Target column contains missing values")