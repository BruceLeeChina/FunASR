FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# 复制应用文件
COPY server.py /app/server.py
COPY hotwords.txt /app/hotwords.txt
COPY templates /app/templates
COPY data /app/data

# 创建临时目录
RUN mkdir -p /app/temp_dir

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]