# Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 更新并安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt并安装依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# 复制应用文件
COPY server.py /app/
COPY hotwords.txt /app/
COPY templates/ /app/templates/
COPY data/ /app/data/

# 创建临时目录和模型目录
RUN mkdir -p /app/temp_dir && chmod 777 /app/temp_dir
RUN mkdir -p /root/.cache/modelscope/hub/models && chmod 777 /root/.cache/modelscope/hub/models

# 暴露端口
EXPOSE 8000

# 使用exec形式启动，确保信号正确传递
ENTRYPOINT ["python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
