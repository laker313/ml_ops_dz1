import grpc
import asyncio
import io
import json
import pandas as pd

from app.api.utils import async_run_in_pool
from app.models.models import MODEL_CLASSES, Models
from app.data_storage.mino.model_storage import (
    delete_model_from_minio,
    save_model_to_minio,
    read_model_from_minio,
    update_model_to_minio
)
from app.data_storage.mino.datasets_storage import read_dataset_from_minio
from app.data_storage.model.model_status import Learning_status
from app.api.dataset_handler import format_validation, read_pd_from_format
from app.api.utils import async_run_in_pool, GLOBAL_THREAD_POOL
import app.proto.grpc_model_handler_pb2 as model_pb2
import app.proto.grpc_model_handler_pb2_grpc as model_pb2_grpc


# -----------------------------
# Вспомогательные функции
# -----------------------------

def create_model(model_name: str, task_type: str = "classifier", **kwargs):
    model_class = MODEL_CLASSES[model_name][task_type]
    return model_class(**kwargs)

def validate_model_name_classifier(model_name, task_type):
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model {model_name} not supported. Available: {list(MODEL_CLASSES.keys())}")
    if task_type not in MODEL_CLASSES[model_name]:
        raise ValueError(f"Task type {task_type} not supported for {model_name}")

def get_model_default_params(model_name: str, task_type: str = 'classifier'):
    model = create_model(model_name, task_type)
    if hasattr(model, 'get_params'):
        return model.get_params()
    elif hasattr(model, 'get_all_params'):
        return model.get_all_params()
    return {k: v for k, v in model.__dict__.items() if not k.startswith('_')}

def validate_hyperparams(model_name, task_type, params):
    defaults = get_model_default_params(model_name, task_type)
    for p in params:
        if p not in defaults:
            raise ValueError(f"Hyperparameter {p} not valid for {model_name} with task {task_type}")


# -----------------------------
# Основной gRPC-сервис
# -----------------------------

class ModelService(model_pb2_grpc.ModelServiceServicer):

    async def Health(self, request, context):
        return model_pb2.HealthResponse(
            status="ok",
            workers=len(GLOBAL_THREAD_POOL._threads),
            queue_size=GLOBAL_THREAD_POOL._work_queue.qsize()
        )

    async def PoolStatus(self, request, context):
        return model_pb2.PoolStatusResponse(
            max_workers=GLOBAL_THREAD_POOL._max_workers,
            active=len(GLOBAL_THREAD_POOL._threads),
            queue=GLOBAL_THREAD_POOL._work_queue.qsize()
        )

    async def GetModelList(self, request, context):
        model_list = [m.value for m in Models]
        return model_pb2.ModelListResponse(model_names=model_list)

    async def CreateAndSaveModel(self, request, context):
        params = json.loads(request.hyperparameters or "{}")
        model = create_model(request.model_name, request.task_type, **params)
        model_id = await async_run_in_pool(save_model_to_minio, model, request.model_name, params)
        return model_pb2.CreateModelResponse(
            status="saved",
            model_name=request.model_name,
            hyperparameters=json.dumps(params),
            model_type=type(model).__name__,
            model_id=model_id
        )

    async def LearnModel(self, request, context):
        resp = await async_run_in_pool(read_model_from_minio, request.model_id)
        model = resp["model"]
        hyperparams = resp["hyperparams"]
        model_name = resp["model_name"]
        dataset_bytes = await async_run_in_pool(read_dataset_from_minio, request.data_id)
        dataset = pd.read_parquet(io.BytesIO(dataset_bytes))

        if "target" not in dataset.columns:
            context.set_details("Dataset must contain 'target' column")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return model_pb2.LearnModelResponse()

        X = dataset.drop(columns=["target"])
        y = dataset["target"]
        await async_run_in_pool(model.fit, X, y)
        await async_run_in_pool(update_model_to_minio, model, request.model_id, model_name, hyperparams, Learning_status.LEARNED)
        return model_pb2.LearnModelResponse(
            status=f"model_{request.model_id}_learned_with_data_{request.data_id}",
            model_type=model_name,
            model_id=request.model_id
        )

    async def DeleteModel(self, request, context):
        await async_run_in_pool(delete_model_from_minio, request.model_id)
        return model_pb2.DeleteModelResponse(
            status="model_deleted",
            model_id=request.model_id
        )
# message PredictionsRequest {
#   string model_id = 1;
#   bytes file_content = 2;     // Содержимое файла для предсказаний
#   string file_name = 3;       // Имя файла
# }
    async def GetPredictionsFromFile(self, request, context):
        resp = await async_run_in_pool(read_model_from_minio, request.model_id)
        model = resp["model"]
        if resp["learning_status"] != Learning_status.LEARNED:
            context.set_details(f"Model {request.model_id} is not learned yet")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return model_pb2.PredictionsResponse()

        file_format = format_validation(request.file_name)
        dataset = await async_run_in_pool(read_pd_from_format, request.file_content, file_format)
        dataset_X = dataset.drop(columns=["target"]) if "target" in dataset.columns else dataset
        predictions = await async_run_in_pool(model.predict, dataset_X)

        # Подготавливаем CSV
        csv_bytes = io.BytesIO()
        pd.DataFrame(predictions, columns=["predictions"]).to_csv(csv_bytes, index=False)
        return model_pb2.PredictionsResponse(
            predictions_csv=csv_bytes.getvalue(),
            file_name="predictions.csv"
        )

    async def GetModel(self, request, context):
        resp = await async_run_in_pool(read_model_from_minio, request.model_id)
        return model_pb2.GetModelResponse(
            model_id=request.model_id,
            model_name=resp["model_name"],
            learning_status=resp["learning_status_str"],
            hyperparams=str(resp["hyperparams"])
        )

    async def GetModelHyperparameters(self, request, context):
        params = get_model_default_params(request.model_name, request.task_type)
        str_params = {k: str(v) for k, v in params.items()}
        return model_pb2.HyperparametersResponse(
            model_name=request.model_name,
            task_type=request.task_type,
            hyperparameters=str_params
        )