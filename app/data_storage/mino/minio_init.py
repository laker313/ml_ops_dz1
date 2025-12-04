from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.data_storage.mino.datasets_storage import DATASETS_BUCKET
from app.data_storage.mino.minio_helper import ensure_bucket_exists, get_minio_client, set_bucket_versioning
from app.data_storage.mino.model_storage import MODEL_BUCKET
from app.clearml.clearml_service import initialize_clearml_task



    # ... остальные стартовые операции ...

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Запускается при старте приложения"""
    

    client = get_minio_client()
    ensure_bucket_exists(client, DATASETS_BUCKET)
    ensure_bucket_exists(client, MODEL_BUCKET)
    set_bucket_versioning(client, DATASETS_BUCKET)
    set_bucket_versioning(client, MODEL_BUCKET)
    initialize_clearml_task()
    yield