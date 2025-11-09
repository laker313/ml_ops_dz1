import logging
import sys
from pathlib import Path
import json
from datetime import datetime
import time
from functools import wraps


class JSONFormatter(logging.Formatter):

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'model_id'):
            log_entry['model_id'] = record.model_id
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        if hasattr(record, 'method_name'):
            log_entry['method_name'] = record.method_name
            
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging():
    """Настройка логгирования для приложения"""
    
    # Создаем логгеры для разных компонентов
    loggers = [
        "handler",  # модели ML
        "model_storage",  # MinIO операции
        "dataset_storage",  # MinIO операции
        "app"  #  общий логгер приложения
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Очищаем существующие обработчики
        logger.handlers.clear()
        
        # Файловый обработчик (JSON)
        file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        
        # Консольный обработчик (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)




def log_endpoint(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger("handler")
        start_time = time.time()
        
        # Извлекаем информацию о запросе (для FastAPI)
        request_info = {}
        for arg in args:
            if hasattr(arg, 'method') and hasattr(arg, 'url'):  # Это request object
                request_info = {
                    'method': arg.method,
                    'url': str(arg.url),
                    'client': f"{arg.client.host}:{arg.client.port}" if arg.client else None
                }
                break
        
        # Логируем начало выполнения
        logger.info(f"Starting {func.__name__}", extra={
            'endpoint': func.__name__,
            'status': 'started',
            'request': request_info,
            'kwargs_keys': list(kwargs.keys())
        })
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Логируем успешное завершение
            logger.info(f"Completed {func.__name__}", extra={
                'endpoint': func.__name__,
                'status': 'completed',
                'execution_time': round(execution_time, 3),
                'response_type': type(result).__name__
            })
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Логируем ошибку
            logger.error(f"Error in {func.__name__}: {str(e)}", extra={
                'endpoint': func.__name__,
                'status': 'error',
                'execution_time': round(execution_time, 3),
                'error_type': type(e).__name__,
                'request': request_info,
                'kwargs_keys': list(kwargs.keys())
            })
            raise
    
    return wrapper





def log_minio_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("model_storage")
        start_time = time.time()
        
        # Извлекаем информацию о модели из аргументов
        model_info = "unknown"
        for arg in args:
            if hasattr(arg, '__class__'):
                model_info = f"{arg.__class__.__name__}"
                # Дополнительная информация для ML моделей
                if hasattr(arg, 'get_params'):
                    try:
                        params = arg.get_params()
                        important_params = {k: v for k, v in params.items() 
                                          if k in ['n_estimators', 'max_depth', 'random_state']}
                        if important_params:
                            model_info += f"(params: {important_params})"
                    except:
                        pass
                break
        
        # Логируем начало выполнения
        logger.info(f"Starting MinIO operation: {func.__name__}", extra={
            'method_name': func.__name__,
            'status': 'started',
            'model_class': model_info,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        })
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Логируем успешное завершение
            logger.info(f"Completed MinIO operation: {func.__name__}", extra={
                'method_name': func.__name__,
                'status': 'completed',
                'execution_time': round(execution_time, 3),
                'model_class': model_info
            })
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Логируем ошибку
            logger.error(f"Error in MinIO operation {func.__name__}: {str(e)}", extra={
                'method_name': func.__name__,
                'status': 'error',
                'execution_time': round(execution_time, 3),
                'error_type': type(e).__name__,
                'model_class': model_info
            })
            raise
    
    return wrapper



def log_minio_dataset(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("dataset_storage")
        start_time = time.time()
        
        # Информация о датасете
        dataset_info = {
            'dataset_id': 'unknown',
            'dataset_name': 'unknown', 
            'file_format': 'unknown',
            'rows': 'unknown',
            'columns': 'unknown',
            'size_bytes': 'unknown'
        }
        
        # Ищем DataFrame в аргументах
        for arg in args:
            if hasattr(arg, '__class__'):
                if hasattr(arg, 'shape'):  # pandas DataFrame
                    dataset_info['rows'] = arg.shape[0]
                    dataset_info['columns'] = arg.shape[1]
                    dataset_info['file_format'] = 'parquet'
                break
        
        # Ищем информацию в kwargs
        dataset_info['dataset_id'] = kwargs.get('dataset_id', 'unknown')
        dataset_info['dataset_name'] = kwargs.get('dataset_name', 'unknown')
        
        # Логируем начало выполнения
        logger.info(f"Starting dataset operation: {func.__name__}", extra={
            'method_name': func.__name__,
            'status': 'started',
            'dataset_id': dataset_info['dataset_id'],
            'dataset_name': dataset_info['dataset_name'],
            'file_format': dataset_info['file_format'],
            'dataset_shape': f"{dataset_info['rows']}x{dataset_info['columns']}",
            'operation_type': 'dataset_storage'
        })
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Если функция возвращает dataset_id - логируем его
            result_info = {}
            if isinstance(result, str) and len(result) > 10:  # предположительно ID
                result_info['created_dataset_id'] = result
            
            # Логируем успешное завершение
            logger.info(f"Completed dataset operation: {func.__name__}", extra={
                'method_name': func.__name__,
                'status': 'completed',
                'execution_time': round(execution_time, 3),
                'dataset_id': dataset_info['dataset_id'],
                **result_info
            })
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Логируем ошибку
            logger.error(f"Error in dataset operation {func.__name__}: {str(e)}", extra={
                'method_name': func.__name__,
                'status': 'error',
                'execution_time': round(execution_time, 3),
                'error_type': type(e).__name__,
                'dataset_id': dataset_info['dataset_id'],
                'dataset_name': dataset_info['dataset_name']
            })
            raise
    
    return wrapper