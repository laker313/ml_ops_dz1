from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter, Response
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
from app.data_storage.mino.datasets_storage import save_dataset_to_minio, update_dataset_to_minio
import io
from app.data_storage.mino.datasets_storage import delete_dataset_from_minio, read_dataset_from_minio
from app.api.utils import async_run_in_pool
from app.logger.logger import log_endpoint

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
@log_endpoint
async def upload_dataset(
    file: UploadFile = File(...)
):
    """Загрузить набор данных в хранилище"""
    
    # Определяем формат
    file_name = file.filename
    file_format = format_validation(file_name)
    content = await file.read()
    try:
        # Читаем файл
        dataset = await async_run_in_pool(read_pd_from_format, content, file_format)
        dataset_validation(dataset)
        
        dataset_id = await async_run_in_pool(save_dataset_to_minio, dataset, file_name)
        
        return {
            "status": "dataset_uploaded", 
            "dataset_id": dataset_id,
            "dataset_name": file_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
    


@router.post("/update_dataset")
@log_endpoint
async def update_dataset(
    file: UploadFile = File(...),
    dataset_id: str = Form(...)
):
    file_name = file.filename
    file_format = format_validation(file_name)
    content = await file.read()
    try:
        # Читаем файл
        dataset = await async_run_in_pool(read_pd_from_format, content, file_format)
        dataset_validation(dataset)
        
        dataset_id = await async_run_in_pool(update_dataset_to_minio,dataset, dataset_id, file_name)
        
        return {
            "status": "dataset_updated", 
            "dataset_id": dataset_id,
            "dataset_name": file_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()









def read_pd_from_format(content, file_format) -> pd.DataFrame:
    file_bytes = io.BytesIO(content)
    dataset = SUPPORTED_FORMATS[file_format]['reader'](file_bytes)
    return dataset

def format_validation(file_name):
    """Загрузить набор данных в хранилище"""
    # Валидация файла
    if not file_name:
        raise ValueError("Filename is required")
    
    # Определяем формат
    file_format = determine_file_format(file_name)
    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format. Supported: {list(SUPPORTED_FORMATS.keys())}")
            
    return file_format
    
@router.post("/download_dataset")
@log_endpoint
async def download_dataset(
    dataset_id: str = Form(...)
):
    """Загрузить набор данных в хранилище"""
    try:
        # Читаем файл
        dataset_bytes = await async_run_in_pool(read_dataset_from_minio, dataset_id)
                # Конвертируем parquet в pandas DataFrame
        dataset = pd.read_parquet(io.BytesIO(dataset_bytes))
        
        # Конвертируем DataFrame в CSV
        csv_bytes = io.BytesIO()
        dataset.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        
        return Response(
            content=csv_bytes.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={dataset_id}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete_dataset")
@log_endpoint
async def delete_dataset(
    dataset_id: str = Form(...)
):

    try:
        
        await async_run_in_pool( delete_dataset_from_minio, dataset_id)
        
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