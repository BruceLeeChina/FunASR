本地测试：
curl.exe -X POST http://localhost:8000/recognition -F "audio=@D:\workspace\PycharmProjects\FunASR\data\automn.wav"
新：
curl.exe -X POST -F "file=@D:\workspace\PycharmProjects\FunASR\data\automn.wav" http://localhost:8000/submit_task
{"code":0,"msg":"任务提交成功","task_id":"f958ddff-da64-11f0-b60b-7cb5664739ba"}

查询结果：
curl.exe http://localhost:8000/get_task_status?task_id=f958ddff-da64-11f0-b60b-7cb5664739ba
curl.exe http://localhost:8000/get_task_result?task_id=f958ddff-da64-11f0-b60b-7cb5664739ba

AI提问规划:
我要基于: server.py 为基础，开发一个支持并发的离线文本识别接口，支持两种方式，一种是server.py已有的上传文件，然后获得结果；另外一种参考 阿里云录音文件识别接口，提交一个文件地址给服务，然后后台读取文件，识别后返回结果。其他要求：
1.过程中支持查询识别状态，支持查询识别结果、支持取消识别、支持删除识别任务、支持查询识别任务列表、支持查询识别任务详情、支持批量操作（每个文件需要在提交和查询的时候对应上）等。
2.给出一个html页面，用于测试接口，文件上传和文件地址提交的方式都需要，支持批量操作；
3.支持：支持单轨和双轨的WAV、MP3、MP4、M4A、WMA、AAC、OGG、AMR、FLAC格式录音文件识别；如果不符合的话，需要转换为模型可以识别的类型。
其他规则参考：录音文件识别接口说明 https://help.aliyun.com/zh/isi/developer-reference/api-reference-2?spm=a2c4g.11186623.help-menu-30413.d_3_2_2_0.4c063a6bG71YKh
调用实例：
https://help.aliyun.com/zh/isi/developer-reference/sdk-for-python-3?spm=a2c4g.11186623.help-menu-30413.d_3_2_2_1_6.70482a8a82egSa




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
http://192.168.21.164:7080/

直接访问：
http://192.168.21.164:8000

接口文档：
http://192.168.21.164:7080/doc/




