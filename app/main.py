from fastapi import FastAPI
from app.api.models_handler import router as models_router
from app.api.dataset_handler import router as dataset_router

# Настройки прямо здесь
PROJECT_NAME = "My Project"
API_V1_STR = "/api/v1"

app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json"
)


app.include_router(models_router, prefix=API_V1_STR)
app.include_router(dataset_router, prefix=API_V1_STR)
# Если роутеров нет, можно оставить основной маршрут
@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
