FROM python:3.11-slim

# Установка инструментов разработки
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgomp1 \
    libgfortran5 \
    libstdc++6 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Копируем и устанавливаем зависимости
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем пользователя для разработки
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME