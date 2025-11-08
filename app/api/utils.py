from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps




GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=40, thread_name_prefix="model_worker")

# def async_run_in_pool(func):
#     """Декоратор для запуска синхронных функций в пуле потоков"""
#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(GLOBAL_THREAD_POOL, lambda: func(*args, **kwargs))
#     return wrapper



async def async_run_in_pool(func, *args, **kwargs):
    """Запускает синхронную функцию в пуле потоков"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        GLOBAL_THREAD_POOL, 
        lambda: func(*args, **kwargs)
    )