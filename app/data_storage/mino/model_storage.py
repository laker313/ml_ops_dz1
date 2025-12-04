import json
from minio import Minio
from minio.error import S3Error
import pickle
import io
from datetime import datetime
from fastapi import HTTPException
from app.data_storage.model.model_status import Learning_status, get_learning_status
from app.data_storage.mino.minio_helper import get_minio_client,ensure_bucket_exists, generate_datatime_uuid4_id
from app.logger.logger import log_minio_model


# Конфигурация MinIO
MINIO_CONFIG = {
    "endpoint": "minio:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "secure": False  # HTTP, не HTTPS
}


MODEL_BUCKET = "ml-models"


@log_minio_model
def save_model_to_minio(model, model_name: str, hyperparameters: dict) -> str:
    """
    Сохранить модель в MinIO и вернуть ID модели
    """
    client = get_minio_client()

    
    # Генерируем уникальный ID для модели
    model_id = generate_datatime_uuid4_id()
    object_name = f"{model_id}.pkl"
    
    try:
        # Сериализуем модель в bytes
        model_bytes = pickle.dumps(model)
        model_stream = io.BytesIO(model_bytes)
        
        # Метаданные
        metadata = {
            "model_name": model_name,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "training_status": Learning_status.NOT_LEARNED.value,
            "hyperparameters": json.dumps(hyperparameters)
        }
        
        # Загружаем в MinIO
        client.put_object(
            bucket_name=MODEL_BUCKET,
            object_name=object_name,
            data=model_stream,
            length=len(model_bytes),
            content_type='application/octet-stream',
            metadata=metadata
        )
        
        print(f"Model saved to MinIO: {object_name}")
        return model_id
        
    except Exception as e:
        raise S3Error(f"Failed to save model to MinIO: {str(e)}") 

@log_minio_model
def read_model_from_minio(model_id: str) -> dict:
    """Загрузить модель из MinIO по ID"""
    try:
        client = get_minio_client()
        object_name = f"{model_id}.pkl"
        
        # Получаем объект из MinIO
        response = client.get_object(MODEL_BUCKET, object_name)
        model_bytes = response.read()
        status_str = response.headers.get("x-amz-meta-training_status")
        hyperparams_str = response.headers.get("x-amz-meta-hyperparameters")

        status = get_learning_status(status_str)
        hyperparams = json.loads(hyperparams_str) if hyperparams_str else {}
        
        # Десериализуем модель
        model = pickle.loads(model_bytes)

        
        return {
            "status": "success",
            "model_id": model_id,
            "model": model,
            "model_name": response.headers.get("x-amz-meta-model_name"),
            "learning_status": status,
            "learning_status_str": status.value,
            "hyperparams": hyperparams
        }

        
    except S3Error as e:
        raise S3Error(f"Model not found: {str(e)}")
    except Exception as e:
        raise S3Error(f"Error loading model: {str(e)}")
    finally:
        response.close()
        response.release_conn()

@log_minio_model
def update_model_to_minio(model, model_id: str, model_name: str, hyperparameters: dict, training_status: Learning_status) -> str:
    """
    Сохранить модель в MinIO и вернуть ID модели
    """
    client = get_minio_client()
    
    # Генерируем уникальный ID для модели
    object_name = f"{model_id}.pkl"
    
    try:
        # Сериализуем модель в bytes
        model_bytes = pickle.dumps(model)
        model_stream = io.BytesIO(model_bytes)
        
        # Метаданные
        metadata = {
            "model_name": model_name,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "training_status": training_status.value,
            "hyperparameters": json.dumps(hyperparameters)
        }
        
        # Загружаем в MinIO
        client.put_object(
            bucket_name=MODEL_BUCKET,
            object_name=object_name,
            data=model_stream,
            length=len(model_bytes),
            content_type='application/octet-stream',
            metadata=metadata
        )
        
        
        print(f"Model saved to MinIO: {object_name}")
        return model_id
        
    except Exception as e:
        raise S3Error(f"Failed to save model to MinIO: {str(e)}") 

@log_minio_model
def delete_model_from_minio(model_id: str) -> bool:
    """Удалить модель из MinIO"""
    client = get_minio_client()
    object_name = f"{model_id}.pkl"
    
    try:
        client.remove_object(MODEL_BUCKET, object_name)
        print(f"Model {object_name} deleted successfully")
        return True
    except Exception as e:
        print(f"Error deleting model: {str(e)}")
        raise S3Error(f"Failed to remove model to MinIO: {str(e)}") 
