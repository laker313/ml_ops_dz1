import json
import pickle
import tempfile
import os
from datetime import datetime
from fastapi import HTTPException
from clearml import Model, StorageManager, Task, OutputModel
import ast

# Импорт ваших исключений и моделей данных
# (Оставляем S3Error, если он используется в try/except блоках выше по стеку,
# хотя по факту это уже не S3 ошибка)
from app.data_storage.model.model_status import Learning_status, get_learning_status
# generate_datatime_uuid4_id и minio_helper больше не нужны для логики, 
# но если они нужны для импортов, можно оставить или убрать.
from app.logger.logger import log_minio_model

# Настройка проекта ClearML (можно вынести в переменные окружения)
CLEARML_PROJECT_NAME = "ML_Models_Storage"

@log_minio_model
def save_model_to_clearml(model, model_name: str, hyperparameters: dict) -> str:
    """
    Сохранить модель в ClearML и вернуть ID модели.
    (Название функции оставлено прежним для сохранения совместимости)
    """
    try:
        # 1. Сериализуем модель во временный файл
        # ClearML работает с файлами, а не потоками байтов напрямую для регистрации моделей
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(model, tmp_file)
            tmp_path = tmp_file.name

        try:
            # 2. Создаем OutputModel (регистрируем новую модель)
            # Если есть активная задача (Task), модель привяжется к ней. 
            # Если нет - создастся "свободная" модель.
            output_model = OutputModel(
                name=model_name,
                tags=[Learning_status.NOT_LEARNED.value],
                framework="Pickle" # Или укажите ваш фреймворк (sklearn, pytorch)
            )
            

            remote_path = StorageManager.upload_file(
                local_file=tmp_path,
                remote_url=''   # ПУСТАЯ строка = используем cloud storage
            )
                        # 3. Прикрепляем файл модели
            output_model.update_weights(weights_filename=remote_path)
            
            # 4. Сохраняем гиперпараметры в конфигурацию модели
            output_model.update_design(config_dict=hyperparameters)
            
            # 5. Сохраняем дополнительные метаданные
            metadata = {
                "created_at": datetime.now().isoformat(),
                "training_status": Learning_status.NOT_LEARNED.value,
                # Можно дублировать имя и ID, хотя они есть в полях ClearML
                "model_name": model_name
            }


            # В ClearML нет прямого dict для метаданных как в S3, 
            # используем comment или labels, но лучше всего - update_design или кастомный dict
            # Для сохранения совместимости при чтении, запишем это в user properties (комментарий или JSON в desc)
            output_model.set_metadata("custom_meta", metadata)

            # Получаем ID, сгенерированный ClearML
            model_id = output_model.id
            
            print(f"Model saved to ClearML. ID: {model_id}, Name: {model_name}")
            return model_id

        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        # Оборачиваем в S3Error для сохранения сигнатуры исключений
        raise Exception(f"Failed to save model to ClearML: {str(e)}")


@log_minio_model
def read_model_from_clearml(model_id: str) -> dict:
    """Загрузить модель из ClearML по ID"""
    try:
        # 1. Получаем объект модели из ClearML
        clearml_model = Model(model_id=model_id)
        
        # 2. Скачиваем файл модели локально
        model_path = clearml_model.get_local_copy()
        if not model_path:
            raise ValueError("Model file path is empty")

        # 3. Десериализуем
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)

        # 4. Извлекаем метаданные
        # Гиперпараметры из design (config)
        hyperparams = clearml_model.config_dict or {}
        
        # Статус из метаданных (которые мы записали в custom_meta)
        custom_meta_raw = clearml_model.get_metadata("custom_meta")

        if isinstance(custom_meta_raw, str):
            try:
                custom_meta = ast.literal_eval(custom_meta_raw)
            except:
                custom_meta = {}
        else:
            custom_meta = custom_meta_raw or {}
        status_str = custom_meta.get("training_status")
        
        # Если в custom_meta нет, попробуем найти в тегах
        if not custom_meta:
            for tag in clearml_model.tags:
                try:
                    # Проверяем, является ли тег валидным статусом
                    get_learning_status(tag) 
                    status_str = tag
                    break
                except:
                    continue

        status = get_learning_status(status_str)
        model_name = clearml_model.name

        return {
            "status": "success",
            "model_id": model_id,
            "model": model_obj,
            "model_name": model_name,
            "learning_status": status,
            "learning_status_str": status.value,
            "hyperparams": hyperparams
        }

    except Exception as e:
        # Оборачиваем в S3Error для сохранения сигнатуры исключений
        raise Exception(f"Failed to save model to ClearML: {str(e)}")


@log_minio_model
def update_model_to_clearml(model, model_id: str, model_name: str,
                            hyperparameters: dict, training_status: Learning_status):
    """
    Реально обновляет существующую модель в ClearML (clearml==1.17.0)
    """
    model_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            model_path = temp_file.name
            pickle.dump(model, temp_file)
            
        # 2. Инициализация временной Task для загрузки файла
        # Создаем 'Ephemeral' (временную) задачу, чтобы она не логировалась
        # или используем Task.init с 'at_exit=True' для автоматического закрытия
        temp_task = Task.create(
        )
        
        # 3. Загрузка файла модели (артефакта)
        # Этот метод использует настройки MinIO из вашей конфигурации
        print(f"Uploading model file from {model_path}...")
        temp_task.upload_artifact(
            name='model_artifact', 
            artifact_object=model_path
            # Добавим теги, чтобы пометить его
        )
        new_artifact_uri = temp_task.artifacts['model_artifact'].uri
        
        # Получаем URI только что загруженного артефакта

        # Закрываем временную задачу, чтобы она завершила работу.
        temp_task.close() 

        # --- 4. Обновление объекта Model ---
        
        clearml_model = Model(model_id=model_id)


        # # 4b. Обновление метаданных
        # clearml_model.update_design(config_dict=hyperparameters)

        # if model_name and model_name != clearml_model.name:
        #     clearml_model.name = model_name

        metadata = {
            "updated_at": datetime.utcnow().isoformat(),
            "training_status": training_status.value,
            "model_name": model_name,
        }
        clearml_model.set_metadata("custom_meta", metadata)
        clearml_model.update_weights(weights_filename=new_artifact_uri)
        
        # 4c. Публикация/Обновление статуса
        clearml_model.publish()

        return model_id
    except Exception as e:
            raise Exception(f"Failed to update model to ClearML: {str(e)}")

    finally:
        # 5. Очистка
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
    

@log_minio_model
def delete_model_from_clearml(model_id: str) -> bool:
    """Удалить модель из ClearML"""
    try:
        # В ClearML удаление модели стирает запись из базы и может удалять файл,
        # если настроено соответствующим образом в сервере.
        Model.remove(model_id)
        print(f"Model {model_id} deleted successfully from ClearML")
        return True
    except Exception as e:
        # Оборачиваем в S3Error для сохранения сигнатуры исключений
        raise Exception(f"Failed to save model to ClearML: {str(e)}")
    
CLEARML_PROJECT_NAME = "ML_Models_Storage"
CLEARML_APP_TASK_NAME = "Model_Storage_API_Backend"

def initialize_clearml_task():
    try:
        # Инициализируем постоянную задачу для всего API
        Task.init(
            project_name=CLEARML_PROJECT_NAME,
            task_name=CLEARML_APP_TASK_NAME,
            task_type=Task.TaskTypes.service
        )
        print("ClearML Service Task Initialized for API Backend.")
    except Exception as e:
        # Ошибка инициализации ClearML не должна останавливать запуск FastAPI
        print(f"Warning: ClearML initialization failed, models may not be logged correctly: {e}")

