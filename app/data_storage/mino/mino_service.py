from minio import Minio
from minio.error import S3Error
import pickle
import io
import uuid
from datetime import datetime
from fastapi import HTTPException



# Конфигурация MinIO
MINIO_CONFIG = {
    "endpoint": "minio:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "secure": False  # HTTP, не HTTPS
}




def load_model_from_minio(model_id: str):
    """Загрузить модель из MinIO по ID"""
    try:
        client = get_minio_client()
        bucket_name = "ml-models"
        object_name = f"{model_id}.pkl"
        
        # Получаем объект из MinIO
        response = client.get_object(bucket_name, object_name)
        model_bytes = response.read()
        
        # Десериализуем модель
        model = pickle.loads(model_bytes)
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": type(model).__name__
        }
        
    except S3Error as e:
        raise HTTPException(404, f"Model not found: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")




def get_minio_client():
    """Создать клиент MinIO"""
    return Minio(
        MINIO_CONFIG["endpoint"],
        access_key=MINIO_CONFIG["access_key"],
        secret_key=MINIO_CONFIG["secret_key"],
        secure=MINIO_CONFIG["secure"]
    )

def ensure_bucket_exists(client, bucket_name: str):
    """Убедиться, что бакет существует"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created")
    except S3Error as e:
        raise HTTPException(500, f"MinIO error: {str(e)}")
    


def save_model_to_minio(model, model_name: str, hyperparameters: dict) -> str:
    """
    Сохранить модель в MinIO и вернуть ID модели
    """
    client = get_minio_client()
    bucket_name = "ml-models"
    
    # Создаем бакет если нужно
    ensure_bucket_exists(client, bucket_name)
    
    # Генерируем уникальный ID для модели
    model_id = str(uuid.uuid4())
    object_name = f"{model_name}/{model_id}.pkl"
    
    try:
        # Сериализуем модель в bytes
        model_bytes = pickle.dumps(model)
        model_stream = io.BytesIO(model_bytes)
        
        # Метаданные
        metadata = {
            "model_name": model_name,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "hyperparameters": str(hyperparameters)
        }
        
        # Загружаем в MinIO
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=model_stream,
            length=len(model_bytes),
            content_type='application/octet-stream',
            metadata=metadata
        )
        
        print(f"Model saved to MinIO: {object_name}")
        return model_id
        
    except S3Error as e:
        raise HTTPException(500, f"Failed to save model to MinIO: {str(e)}") 