from minio.error import S3Error
from minio import Minio
import uuid
from datetime import datetime
from fastapi import HTTPException

MINIO_CONFIG = {
    "endpoint": "minio:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "secure": False
}


def get_minio_client():
    """Создать клиент MinIO"""
    return Minio(
        MINIO_CONFIG["endpoint"],
        access_key=MINIO_CONFIG["access_key"],
        secret_key=MINIO_CONFIG["secret_key"],
        secure=MINIO_CONFIG["secure"]
    )


def ensure_bucket_exists(client: Minio, bucket_name: str):
    """Убедиться, что бакет существует"""
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created")
    except S3Error as e:
        raise HTTPException(500, f"MinIO error: {str(e)}")
    

def generate_datatime_uuid4_id():
    base_id = uuid.uuid4()
    timestamp = datetime.now().isoformat()
    return f"{base_id}_{timestamp}"