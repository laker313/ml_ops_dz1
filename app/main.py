from fastapi import FastAPI

# Настройки прямо здесь
PROJECT_NAME = "My Project"
API_V1_STR = "/api/v1"

app = FastAPI(
    title=PROJECT_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json"
)

# Если роутеров нет, можно оставить основной маршрут
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# @app.get("/123/")
# def read_root():
#     return {"message": "Hello World123"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
