import io
import grpc 
from concurrent import futures
import asyncio

import pandas as pd


from app.api.utils import async_run_in_pool

from app.api.dataset_handler import (
    format_validation, 
    read_pd_from_format, 
    dataset_validation
)
from app.data_storage.mino.datasets_storage import delete_dataset_from_minio, read_dataset_from_minio, save_dataset_to_minio, update_dataset_to_minio
import app.proto.grpc_dataset_handler_pb2 as dataset_pb2
import app.proto.grpc_dataset_handler_pb2_grpc as dataset_pb2_grpc



class DatasetService(dataset_pb2_grpc.DataServiceServicer):
    # ------------------------------
    # Upload dataset
    # ------------------------------
    async def UploadDataset(self, request, context):
        file_name = request.file_name
        file_format = format_validation(file_name)
        content = request.file_content

        try:
            dataset = await async_run_in_pool(read_pd_from_format, content, file_format)
            dataset_validation(dataset)
            dataset_id = await async_run_in_pool(save_dataset_to_minio, dataset, file_name)

            return dataset_pb2.DatasetOperationResponse(
                status="dataset_uploaded",
                dataset_id=dataset_id,
                dataset_name=file_name
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    # ------------------------------
    # Update dataset
    # ------------------------------
    async def UpdateDataset(self, request, context):
        file_name = request.file_name
        dataset_id = request.dataset_id
        file_format = format_validation(file_name)
        content = request.file_content

        try:
            dataset = await async_run_in_pool(read_pd_from_format, content, file_format)
            dataset_validation(dataset)
            dataset_id = await async_run_in_pool(update_dataset_to_minio, dataset, dataset_id, file_name)

            return dataset_pb2.DatasetOperationResponse(
                status="dataset_updated",
                dataset_id=dataset_id,
                dataset_name=file_name
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    # ------------------------------
    # Download dataset
    # ------------------------------
    async def DownloadDataset(self, request, context):
        dataset_id = request.dataset_id

        try:
            dataset_bytes = await async_run_in_pool(read_dataset_from_minio, dataset_id)

            # читаем parquet → pandas
            dataset = pd.read_parquet(io.BytesIO(dataset_bytes))

            # преобразуем в CSV для передачи
            csv_bytes = io.BytesIO()
            dataset.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)

            return dataset_pb2.DownloadDatasetResponse(
                dataset_content=csv_bytes.getvalue(),
                file_name=f"{dataset_id}.csv"
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    # ------------------------------
    # Delete dataset
    # ------------------------------
    async def DeleteDataset(self, request, context):
        dataset_id = request.dataset_id

        try:
            await async_run_in_pool(delete_dataset_from_minio, dataset_id)

            return dataset_pb2.DatasetOperationResponse(
                status="dataset_deleted",
                dataset_id=dataset_id
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))