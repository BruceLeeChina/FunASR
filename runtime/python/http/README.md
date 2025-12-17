# FunASR 离线文本识别接口

## 功能介绍

基于FunASR开发的支持并发的离线文本识别接口，支持以下功能：

1. **两种识别方式**：
   - 文件上传方式
   - 文件地址提交方式（支持HTTP/HTTPS协议）

2. **完整的任务管理**：
   - 查询识别状态
   - 查询识别结果
   - 取消识别
   - 删除识别任务
   - 查询识别任务列表
   - 查询识别任务详情
   - 支持批量操作

3. **音频格式支持**：
   - 支持单轨和双轨的WAV、MP3、MP4、M4A、WMA、AAC、OGG、AMR、FLAC格式
   - 自动将不符合要求的音频格式转换为模型可识别的格式

4. **并发处理**：
   - 支持配置最大并发任务数
   - 任务队列管理，避免资源耗尽

5. **内置测试页面**：
   - 提供直观的HTML测试页面，支持所有功能的测试

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python server.py --host 0.0.0.0 --port 8000
```

### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| --host | str | 0.0.0.0 | 服务器监听地址 |
| --port | int | 8000 | 服务器监听端口 |
| --asr_model | str | paraformer-zh | ASR模型名称 |
| --vad_model | str | fsmn-vad | VAD模型名称 |
| --punc_model | str | ct-punc-c | 标点模型名称 |
| --ngpu | int | 1 | GPU数量（0表示使用CPU） |
| --device | str | cuda | 设备类型（cuda或cpu） |
| --ncpu | int | 4 | CPU核心数量 |
| --hotword_path | str | hotwords.txt | 热词文件路径 |
| --certfile | str | None | SSL证书文件路径 |
| --keyfile | str | None | SSL私钥文件路径 |
| --temp_dir | str | temp_dir/ | 临时文件目录 |
| --max_concurrent_tasks | int | 10 | 最大并发任务数 |

## API接口文档

### 1. 提交识别任务

```
POST /submit_task
```

#### 请求参数

**文件上传方式：**
- `file`: 音频文件（multipart/form-data格式）

**文件地址方式：**
- `file_url`: 音频文件URL（form-data格式）
- `file_name`: 文件名（可选，form-data格式）

#### 返回结果

```json
{
  "code": 0,
  "msg": "任务提交成功",
  "task_id": "任务ID"
}
```

### 2. 查询任务状态

```
GET /get_task_status?task_id=任务ID
```

#### 返回结果

```json
{
  "code": 0,
  "task_id": "任务ID",
  "status": "pending/processing/completed/failed/canceled",
  "progress": 0.5,
  "updated_time": 1620000000
}
```

### 3. 查询任务结果

```
GET /get_task_result?task_id=任务ID
```

#### 返回结果

```json
{
  "code": 0,
  "status": "completed",
  "result": {
    "text": "识别结果文本",
    "sentences": [
      {
        "text": "句子1",
        "start": 0.0,
        "end": 1.5
      },
      {
        "text": "句子2",
        "start": 1.5,
        "end": 3.0
      }
    ],
    "code": 0
  }
}
```

### 4. 取消任务

```
POST /cancel_task
```

#### 请求参数

- `task_id`: 任务ID（form-data格式）

#### 返回结果

```json
{
  "code": 0,
  "msg": "任务取消成功"
}
```

### 5. 删除任务

```
POST /delete_task
```

#### 请求参数

- `task_id`: 任务ID（form-data格式）

#### 返回结果

```json
{
  "code": 0,
  "msg": "任务删除成功"
}
```

### 6. 查询任务列表

```
GET /list_tasks?page=1&page_size=10&status=completed
```

#### 请求参数

- `page`: 页码（默认1）
- `page_size`: 每页数量（默认10，最大100）
- `status`: 任务状态过滤（可选）

#### 返回结果

```json
{
  "code": 0,
  "total": 100,
  "page": 1,
  "page_size": 10,
  "tasks": [
    {
      "task_id": "任务ID",
      "task_type": "file_upload/file_url",
      "file_name": "文件名",
      "status": "completed",
      "created_time": 1620000000,
      "updated_time": 1620000100
    }
  ]
}
```

### 7. 查询任务详情

```
GET /get_task_details?task_id=任务ID
```

#### 返回结果

```json
{
  "code": 0,
  "task_details": {
    "task_id": "任务ID",
    "task_type": "file_upload/file_url",
    "file_path": "本地文件路径",
    "file_url": "文件URL",
    "file_name": "文件名",
    "status": "completed",
    "progress": 1.0,
    "error_msg": "",
    "created_time": 1620000000,
    "updated_time": 1620000100,
    "result": {
      "text": "识别结果文本",
      "sentences": [],
      "code": 0
    }
  }
}
```

### 8. 批量操作

```
POST /batch_operation
```

#### 请求参数

**批量提交：**
- `operation`: submit
- `file_urls`: 文件URL列表（逗号分隔）
- `file_names`: 文件名列表（逗号分隔，可选）

**批量取消：**
- `operation`: cancel
- `task_ids`: 任务ID列表（逗号分隔）

**批量删除：**
- `operation`: delete
- `task_ids`: 任务ID列表（逗号分隔）

#### 返回结果

```json
{
  "code": 0,
  "msg": "批量操作完成",
  "results": [
    {
      "code": 0,
      "msg": "任务提交成功",
      "task_id": "任务ID"
    }
  ]
}
```

## 测试页面

服务启动后，可以通过浏览器访问 `http://host:port/` 打开内置的测试页面，支持以下功能：

1. **单任务提交**：
   - 文件上传方式
   - 文件地址方式

2. **任务管理**：
   - 查询任务状态
   - 查询任务结果
   - 取消任务
   - 删除任务

3. **批量操作**：
   - 批量提交URL
   - 批量取消任务
   - 批量删除任务

4. **任务列表**：
   - 查看所有任务的状态和信息

## 示例代码

### Python 示例

```python
import requests
import json

# 1. 文件上传方式
def submit_file_upload(file_path):
    url = "http://localhost:8000/submit_task"
    files = {
        'file': open(file_path, 'rb')
    }
    response = requests.post(url, files=files)
    return response.json()

# 2. 文件地址方式
def submit_file_url(file_url, file_name=None):
    url = "http://localhost:8000/submit_task"
    data = {
        'file_url': file_url
    }
    if file_name:
        data['file_name'] = file_name
    response = requests.post(url, data=data)
    return response.json()

# 3. 查询任务状态
def get_task_status(task_id):
    url = f"http://localhost:8000/get_task_status?task_id={task_id}"
    response = requests.get(url)
    return response.json()

# 4. 查询任务结果
def get_task_result(task_id):
    url = f"http://localhost:8000/get_task_result?task_id={task_id}"
    response = requests.get(url)
    return response.json()

# 5. 取消任务
def cancel_task(task_id):
    url = "http://localhost:8000/cancel_task"
    data = {
        'task_id': task_id
    }
    response = requests.post(url, data=data)
    return response.json()

# 6. 删除任务
def delete_task(task_id):
    url = "http://localhost:8000/delete_task"
    data = {
        'task_id': task_id
    }
    response = requests.post(url, data=data)
    return response.json()

# 7. 批量提交
def batch_submit(file_urls, file_names=None):
    url = "http://localhost:8000/batch_operation"
    data = {
        'operation': 'submit',
        'file_urls': ','.join(file_urls)
    }
    if file_names:
        data['file_names'] = ','.join(file_names)
    response = requests.post(url, data=data)
    return response.json()
```

### 命令行示例

```bash
# 文件上传
curl -X POST -F "file=@test.wav" http://localhost:8000/submit_task

# 文件地址
curl -X POST -d "file_url=http://example.com/test.wav&file_name=test.wav" http://localhost:8000/submit_task

# 查询状态
curl http://localhost:8000/get_task_status?task_id=task123

# 查询结果
curl http://localhost:8000/get_task_result?task_id=task123

# 取消任务
curl -X POST -d "task_id=task123" http://localhost:8000/cancel_task

# 删除任务
curl -X POST -d "task_id=task123" http://localhost:8000/delete_task

# 批量提交
curl -X POST -d "operation=submit&file_urls=http://example.com/test1.wav,http://example.com/test2.wav&file_names=test1.wav,test2.wav" http://localhost:8000/batch_operation
```

## 注意事项

1. **音频格式转换**：
   - 系统会自动将所有支持的音频格式转换为模型可识别的PCM格式
   - 确保FFmpeg已正确安装在系统中

2. **并发控制**：
   - 根据服务器配置调整`--max_concurrent_tasks`参数
   - 建议根据CPU/GPU资源合理设置并发数

3. **文件大小限制**：
   - 建议单个音频文件大小不超过100MB
   - 大文件会占用更多的处理时间和资源

4. **任务存储**：
   - 任务信息存储在内存数据库中，服务重启后数据会丢失
   - 如需持久化存储，可修改代码使用文件数据库

5. **安全考虑**：
   - 生产环境建议使用HTTPS协议（通过`--certfile`和`--keyfile`参数配置）
   - 考虑添加API密钥认证机制

## 故障排除

1. **FFmpeg错误**：
   - 确保FFmpeg已正确安装
   - 检查FFmpeg命令是否可以正常执行

2. **任务一直处于pending状态**：
   - 检查服务器资源使用情况
   - 检查`max_concurrent_tasks`参数设置是否合理

3. **识别结果为空**：
   - 检查音频文件是否包含有效语音
   - 检查音频格式是否支持
   - 检查音频音量是否正常

4. **下载文件失败**：
   - 检查文件URL是否可访问
   - 检查网络连接是否正常
   - 检查服务器是否有权限访问该URL

## 依赖安装

```bash
pip install -r requirements.txt
```

## 版本更新

### v1.0.0
- 初始版本发布
- 支持文件上传和文件地址两种提交方式
- 完整的任务管理功能
- 支持批量操作
- 内置测试页面
- 并发处理支持
# Service with http-python

## Server

1. Install requirements

```shell
cd funasr/runtime/python/http
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. Start server

```shell
python server.py --port 8000
```

More parameters:
```shell
python server.py \
--host [host ip] \
--port [server port] \
--asr_model [asr model_name] \
--vad_model [vad model_name] \
--punc_model [punc model_name] \
--device [cuda or cpu] \
--ngpu [0 or 1] \
--ncpu [1 or 4] \
--hotword_path [path of hot word txt] \
--certfile [path of certfile for ssl] \
--keyfile [path of keyfile for ssl] \
--temp_dir [upload file temp dir] 
```

## Client

```shell
# get test audio file
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
python client.py --host=127.0.0.1 --port=8000 --audio_path=asr_example_zh.wav
```

More parameters:
```shell
python server.py \
--host [sever ip] \
--port [sever port] \
--audio_path [use audio path] 
```


## 支持多进程

方法是启动多个`server.py`，然后通过Nginx的负载均衡分发请求，达到支持多用户同时连效果，处理方式如下，默认您已经安装了Nginx，没安装的请参考[官方安装教程](https://nginx.org/en/linux_packages.html#Ubuntu)。

配置Nginx。
```shell
sudo cp -f asr_nginx.conf /etc/nginx/nginx.conf
sudo service nginx reload
```

然后使用脚本启动多个服务，每个服务的端口号不一样。
```shell
sudo chmod +x start_server.sh
./start_server.sh
```

**说明：** 默认是3个进程，如果需要修改，首先修改`start_server.sh`的最后那部分，可以添加启动数量。然后修改`asr_nginx.conf`配置文件的`upstream backend`部分，增加新启动的服务，可以使其他服务器的服务。
