import asyncio
import grpc
import uvicorn

from fastapi import FastAPI


from app.api.models_handler import router as models_router
from app.api.dataset_handler import router as dataset_router
from app.logger.logger import setup_logging

# gRPC сервисы
from app.proto import grpc_dataset_handler_pb2_grpc, grpc_model_handler_pb2_grpc
from app.grpc.dataset_grpc_handler import DatasetService
from app.grpc.model_grpc_handler import ModelService

# Настройки проекта
PROJECT_NAME = "ML_OPS_DZ1"
API_V1_STR = "/api/v1"

setup_logging()

# --- Создаем приложение FastAPI ---
app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json",
    max_upload_size=500 * 1024 * 1024,  # 500MB
    max_response_size=500 * 1024 * 1024  # 500MB
)

# Подключаем роутеры
app.include_router(models_router, prefix=API_V1_STR)
app.include_router(dataset_router, prefix=API_V1_STR)


# --- gRPC сервер ---
async def serve_grpc() -> None:
    print(f"✅ зашли в serve_grpc")
    server = grpc.aio.server()
    grpc_dataset_handler_pb2_grpc.add_DataServiceServicer_to_server(DatasetService(), server)
    print(f"✅ добавили DatasetService")
    grpc_model_handler_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    print(f"✅ добавили ModelService")

    listen_addr = "0.0.0.0:50051"
    server.add_insecure_port(listen_addr)
    print(f"✅ gRPC server started on {listen_addr}")

    await server.start()
    await server.wait_for_termination()


# --- FastAPI сервер ---
async def serve_fastapi() -> None:
    """Запуск FastAPI через uvicorn в асинхронном виде."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    print("✅ FastAPI started on http://0.0.0.0:8000")
    await server.serve()


# --- Основная точка входа ---
async def main():
    """Запуск FastAPI и gRPC одновременно."""
    
    await asyncio.gather(
        serve_fastapi(),
        serve_grpc(),
    )


if __name__ == "__main__":
    asyncio.run(main())
