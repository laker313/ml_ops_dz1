from fastapi import FastAPI
from app.api.models_handler import router as models_router
from app.api.dataset_handler import router as dataset_router

# Настройки прямо здесь
PROJECT_NAME = "ML_OPS_DZ1"
API_V1_STR = "/api/v1"

app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json",
    max_upload_size=500 * 1024 * 1024,  # 500MB
    max_response_size=500 * 1024 * 1024  # 500MB
)


app.include_router(models_router, prefix=API_V1_STR)
app.include_router(dataset_router, prefix=API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
