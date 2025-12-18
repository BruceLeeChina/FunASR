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

## 接口文档

完整的API接口文档请参见单独的文档文件：

- **API Documentation**: [asr_offline_api_doc.md](doc/asr_offline_api_doc.md)