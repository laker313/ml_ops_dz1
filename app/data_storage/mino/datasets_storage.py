from minio.error import S3Error
import pickle
from datetime import datetime
from fastapi import HTTPException
import pandas as pd
from app.data_storage.mino.minio_helper import generate_datatime_uuid4_id,get_minio_client, ensure_bucket_exists
import io



DATASETS_BUCKET = "ml-datasets"

def save_dataset_to_minio(dataset:pd.DataFrame, dataset_name: str) -> str:
    """
    Сохранить данные в MinIO и вернуть ID данных
    """
    client = get_minio_client()

    
    # Создаем бакет если нужно
    ensure_bucket_exists(client, DATASETS_BUCKET)
    
    # Генерируем уникальный ID для датасета
    dataset_id = generate_datatime_uuid4_id()
    object_name = f"{dataset_id}.parquet"

    try:
        # Сериализуем датасет в bytes
        dataset_bytes = io.BytesIO()
        dataset.to_parquet(dataset_bytes, engine='pyarrow', index=False)
        
        # ПОЛУЧАЕМ РАЗМЕР ДО seek(0)
        file_size = dataset_bytes.tell()  # текущая позиция = размер файла
        dataset_bytes.seek(0)  # перематываем для чтения
        
        # Метаданные
        metadata = {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "rows": str(len(dataset)),
            "columns": str(len(dataset.columns)),
            "created_at": datetime.now().isoformat(),
            "size_bytes": str(file_size)
        }
        
        # Загружаем в MinIO
        client.put_object(
            bucket_name=DATASETS_BUCKET,
            object_name=object_name,
            data=dataset_bytes,
            length=file_size,
            content_type='application/parquet',
            metadata=metadata
        )
        
        print(f"dataset saved to MinIO: {object_name}")
        return dataset_id
        
    except S3Error as e:
        raise HTTPException(500, f"Failed to save dataset to MinIO: {str(e)}")


def read_dataset_from_minio(dataset_id: str) -> bytes:

    """Загрузить данные из MinIO по ID"""
    client = get_minio_client()
    response  = client.get_object(DATASETS_BUCKET, f"{dataset_id}.parquet")
    
    try:
        return response.read()
    
    finally:
        response.close()
        response.release_conn()


def update_dataset_to_minio(dataset:pd.DataFrame, dataset_id: str, dataset_name: str) -> str:
    """
    Сохранить данные в MinIO и вернуть ID данных
    """
    client = get_minio_client()

    
    # Создаем бакет если нужно
    ensure_bucket_exists(client, DATASETS_BUCKET)
    
    # Генерируем уникальный ID для датасета
    object_name = f"{dataset_id}.parquet"

    try:
        # Сериализуем датасет в bytes
        dataset_bytes = io.BytesIO()
        dataset.to_parquet(dataset_bytes, engine='pyarrow', index=False)
        
        # ПОЛУЧАЕМ РАЗМЕР ДО seek(0)
        file_size = dataset_bytes.tell()  # текущая позиция = размер файла
        dataset_bytes.seek(0)  # перематываем для чтения
        
        # Метаданные
        metadata = {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "rows": str(len(dataset)),
            "columns": str(len(dataset.columns)),
            "created_at": datetime.now().isoformat(),
            "size_bytes": str(file_size)
        }
        
        # Загружаем в MinIO
        client.put_object(
            bucket_name=DATASETS_BUCKET,
            object_name=object_name,
            data=dataset_bytes,
            length=file_size,
            content_type='application/parquet',
            metadata=metadata
        )
        
        print(f"dataset saved to MinIO: {object_name}")
        return dataset_id
        
    except S3Error as e:
        raise HTTPException(500, f"Failed to save dataset to MinIO: {str(e)}")


def delete_dataset_from_minio(dataset_id: str) -> bool:
    """Удалить данные из MinIO"""
    client = get_minio_client()
    object_name = f"{dataset_id}.parquet"
    
    try:
        client.remove_object(DATASETS_BUCKET, object_name)
        print(f"dataset {object_name} deleted successfully")
        return True
    except S3Error as e:
         raise HTTPException(500, f"Failed to delete dataset to MinIO: {str(e)}")