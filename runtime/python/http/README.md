打包构建：
# 创建项目目录
sudo mkdir -p /opt/funasr-offline

# 进入项目目录
cd /opt/funasr-offline

# 将项目文件复制到此目录中，确保包含以下文件：
# - Dockerfile
# - docker-compose.yml
# - server.py
# - requirements.txt
# - hotwords.txt
# - templates/ (目录)
# - data/ (目录)

# 设置适当的权限
sudo chown -R $USER:$USER /opt/funasr-offline

# 进入项目目录
cd /opt/funasr-offline

# 构建Docker镜像
docker build -t funasr-offline:1.0.0 .


部署：
# 进入项目目录
cd /opt/funasr-offline

# 运行服务（包括Nginx反向代理）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f


ASR离线识别地址：
Nginx访问：
http://192.168.21.130:7080/

直接访问：
http://192.168.21.130:8000

接口文档：
http://192.168.21.130:7080/doc/
