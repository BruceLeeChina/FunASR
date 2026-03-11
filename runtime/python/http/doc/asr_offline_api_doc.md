# FunASR HTTP API 接口文档

## 1. 服务概述

FunASR HTTP服务是基于阿里达摩院FunASR模型的语音识别服务接口，提供了离线音频文件的语音识别功能，支持文件上传和URL提交两种方式，并提供了丰富的任务管理接口。

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 启动服务

```bash
python server.py --host 0.0.0.0 --port 8002 --device cpu --ngpu 0
```

### 2.3 主要参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --host | str | 0.0.0.0 | 服务监听地址 |
| --port | int | 8000 | 服务监听端口 |
| --asr_model | str | paraformer-zh | ASR模型名称 |
| --vad_model | str | fsmn-vad | VAD模型名称 |
| --punc_model | str | ct-punc-c | 标点模型名称 |
| --device | str | cuda | 设备类型（cuda/cpu） |
| --ngpu | int | 1 | GPU数量（0表示使用CPU） |
| --ncpu | int | 4 | CPU核心数 |
| --hotword_path | str | hotwords.txt | 热词文件路径 |
| --temp_dir | str | temp_dir/ | 临时文件目录 |
| --max_concurrent_tasks | int | 10 | 最大并发任务数 |
| --db_pool_size | int | 10 | 数据库连接池大小 |
| --asr_thread_pool_size | int | 4 | ASR处理线程池大小 |

## 2.1 识别模式说明

FunASR支持两种识别模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| default | 标准语音识别模式 | 单人语音、音频转文字 |
| meeting | 会议识别模式 | 多人对话、识别说话人信息 |

**会议识别模式特性：**
- 支持多人对话场景
- 自动识别说话人信息（Speaker 1, Speaker 2, ...）
- 返回对话格式的识别结果
- 包含每段语音的时间戳信息

## 3. 识别模式使用说明

### 3.1 标准识别模式 (default)

标准识别模式适用于单人语音识别，提供基本的语音转文字功能。

**请求参数：**
- `recognition_mode`: "default" (可选，默认为此模式)

**返回结果格式：**
```json
{
  "code": 0,
  "text": "识别结果文本",
  "sentences": [
    {
      "text": "第一句",
      "start": 0,
      "end": 5
    },
    ...
  ]
}
```

### 3.2 会议识别模式 (meeting)

会议识别模式专为多人对话场景设计，能够识别不同说话人并返回对话格式的结果。

**请求参数：**
- `recognition_mode`: "meeting"

**返回结果格式：**
```json
{
  "code": 0,
  "dialogue": [
    {
      "speaker": "Speaker 1",
      "text": "大家好，今天我们讨论一下项目进展",
      "start_time": 0.0,
      "end_time": 3.5
    },
    {
      "speaker": "Speaker 2", 
      "text": "好的，我来汇报一下技术方案",
      "start_time": 3.8,
      "end_time": 7.2
    },
    ...
  ]
}
```

### 3.3 识别模式选择建议

- **单人语音** → 使用 `default` 模式
- **多人会议** → 使用 `meeting` 模式
- **播客音频** → 使用 `meeting` 模式（可识别不同嘉宾）
- **演讲录音** → 使用 `default` 模式
- **对话访谈** → 使用 `meeting` 模式

## 4. API接口列表

### 3.1 任务提交接口

#### 3.1.1 单任务提交

**接口地址**：`POST /submit_task`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| file | file | 否 | 音频文件，与file_url二选一 |
| file_url | string | 否 | 音频文件URL，与file二选一 |
| file_name | string | 否 | 文件名 |
| callback_url | string | 否 | 任务完成后回调URL |
| app_id | string | 否 | 应用ID |
| biz_type | string | 否 | 业务类型 |
| biz_unique_id | string | 否 | 业务唯一ID |
| recognition_mode | string | 否 | 识别模式：default/meeting，默认default |

**返回结果**：

```json
{
  "code": 0,
  "msg": "任务提交成功",
  "task_id": "task_id_value"
}
```

#### 3.1.2 批量任务提交

**接口地址**：`POST /batch_operation`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| operation | string | 是 | 操作类型：submit/cancel/delete |
| file_list | string | 否 | 文件列表，JSON格式：[{"file_url":"xxx","file_name":"xxx","callback_url":"xx"}] |
| file_urls | string | 否 | 文件URL列表，逗号分隔 |
| file_names | string | 否 | 文件名列表，逗号分隔 |
| callback_url | string | 否 | 任务完成后回调URL |
| app_id | string | 否 | 应用ID |
| biz_type | string | 否 | 业务类型 |
| biz_unique_id | string | 否 | 业务唯一ID |
| recognition_mode | string | 否 | 识别模式：default/meeting，默认default |

**返回结果**：

```json
{
  "code": 0,
  "msg": "批量操作成功",
  "results": [
    {
      "code": 0,
      "task_id": "task_id_value"
    },
    ...
  ]
}
```

### 3.2 任务查询接口

#### 3.2.1 查询任务状态

**接口地址**：`GET /get_task_status`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | string | 是 | 任务ID |

**返回结果**：

```json
{
  "code": 0,
  "task_id": "task_id_value",
  "status": "pending",
  "progress": 0,
  "updated_time": 1620000000,
  "callback_status": "pending",
  "recognition_mode": "meeting"
}
```

#### 3.2.2 查询任务结果

**接口地址**：`GET /get_task_result`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | string | 是 | 任务ID |

**返回结果**：

```json
{
  "code": 0,
  "status": "completed",
  "result": {
    "text": "识别结果文本",
    "sentences": [
      {
        "text": "第一句",
        "start": 0,
        "end": 5
      },
      ...
    ],
    "code": 0
  },
  "callback_status": "success",
  "recognition_mode": "default"
}
```

**会议模式返回结果示例：**
```json
{
  "code": 0,
  "status": "completed",
  "result": {
    "dialogue": [
      {
        "speaker": "Speaker 1",
        "text": "大家好，今天我们讨论一下项目进展",
        "start_time": 0.0,
        "end_time": 3.5
      },
      {
        "speaker": "Speaker 2", 
        "text": "好的，我来汇报一下技术方案",
        "start_time": 3.8,
        "end_time": 7.2
      }
    ],
    "code": 0
  },
  "callback_status": "success",
  "recognition_mode": "meeting"
}
```

#### 3.2.3 查询任务详情

**接口地址**：`GET /get_task_details`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | string | 是 | 任务ID |

**返回结果**：

```json
{
  "code": 0,
  "msg": "查询任务详情成功",
  "task": {
    "task_id": "task_id_value",
    "task_type": "file_upload",
    "file_path": "path/to/file",
    "file_url": null,
    "file_name": "filename.wav",
    "status": "completed",
    "progress": 1.0,
    "result": {
      "text": "识别结果文本",
      "sentences": [],
      "code": 0
    },
    "error_message": null,
    "created_time": 1620000000,
    "updated_time": 1620000000,
    "callback_status": "success",
    "recognition_mode": "default"
  }
}
```

#### 3.2.4 查询任务列表

**接口地址**：`GET /list_tasks`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| page | int | 否 | 页码，默认1 |
| page_size | int | 否 | 每页数量，默认10，最大100 |
| status | string | 否 | 任务状态过滤 |

**返回结果**：

```json
{
  "code": 0,
  "msg": "查询任务列表成功",
  "tasks": [
    {
      "task_id": "task_id_value",
      "task_type": "file_upload",
      "file_name": "filename.wav",
      "status": "completed",
      "created_time": 1620000000,
      "updated_time": 1620000000,
      "callback_status": "success",
      "recognition_mode": "meeting"
    },
    ...
  ],
  "total": 100,
  "page": 1,
  "limit": 10
}
```

### 3.3 批量查询接口

#### 3.3.1 批量查询任务状态

**接口地址**：`POST /batch_get_task_status`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_ids | string | 是 | 任务ID列表，逗号分隔 |

**返回结果**：

```json
{
  "code": 0,
  "msg": "批量查询任务状态完成",
  "results": [
    {
      "code": 0,
      "task_id": "task_id_value",
      "status": "completed",
      "progress": 1.0,
      "updated_time": 1620000000,
      "callback_status": "success",
      "recognition_mode": "default"
    },
    ...
  ]
}
```

#### 3.3.2 批量查询任务结果

**接口地址**：`POST /batch_get_task_result`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_ids | string | 是 | 任务ID列表，逗号分隔 |

**返回结果**：

```json
{
  "code": 0,
  "msg": "批量查询任务结果完成",
  "results": [
    {
      "code": 0,
      "task_id": "task_id_value",
      "status": "completed",
      "result": {
        "text": "识别结果文本",
        "sentences": [],
        "code": 0
      },
      "callback_status": "success",
      "recognition_mode": "meeting"
    },
    ...
  ]
}
```

#### 3.3.3 批量查询任务详情

**接口地址**：`POST /batch_get_task_details`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_ids | string | 是 | 任务ID列表，逗号分隔 |

**返回结果**：

```json
{
  "code": 0,
  "msg": "批量查询任务详情完成",
  "results": [
    {
      "code": 0,
      "task_details": {
        "task_id": "task_id_value",
        "task_type": "file_upload",
        "file_path": "path/to/file",
        "file_url": null,
        "file_name": "filename.wav",
        "status": "completed",
        "progress": 1.0,
        "error_msg": null,
        "created_time": 1620000000,
        "updated_time": 1620000000,
        "result": {
          "text": "识别结果文本",
          "sentences": [],
          "code": 0
        }
      }
    },
    ...
  ]
}
```

### 3.4 任务操作接口

#### 3.4.1 取消任务

**接口地址**：`POST /cancel_task`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | string | 是 | 任务ID |

**返回结果**：

```json
{
  "code": 0,
  "msg": "任务取消成功"
}
```

#### 3.4.2 删除任务

**接口地址**：`POST /delete_task`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | string | 是 | 任务ID |

**返回结果**：

```json
{
  "code": 0,
  "msg": "任务删除成功"
}
```

### 3.5 测试接口

#### 3.5.1 模拟回调接口

**接口地址**：`POST /mock_callback`

**请求参数**：

```json
{
  "task_id": "task_id_value",
  "status": "completed",
  "timestamp": 1620000000,
  "result": {
    "text": "识别结果文本",
    "sentences": [],
    "code": 0
  }
}
```

**返回结果**：

```json
{
  "code": 0,
  "msg": "回调接收成功",
  "received_data": { ... }
}
```

### 3.6 高级查询接口

#### 3.6.1 根据业务ID查询任务

**接口地址**：`GET /get_task_by_biz_id`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| biz_unique_id | string | 否 | 业务唯一ID |
| app_id | string | 否 | 应用ID |
| recognition_mode | string | 否 | 识别模式 |

**注意：** 至少需要提供 `biz_unique_id` 或 `app_id` 中的一个参数

**返回结果**：

```json
{
  "code": 0,
  "msg": "查询任务详情成功",
  "task": {
    "task_id": "task_id_value",
    "task_type": "file_upload",
    "file_path": "path/to/file",
    "file_url": null,
    "file_name": "filename.wav",
    "status": "completed",
    "progress": 1.0,
    "result": {
      "text": "识别结果文本",
      "sentences": [],
      "code": 0
    },
    "error_message": null,
    "created_time": 1620000000,
    "updated_time": 1620000000,
    "callback_status": "success",
    "app_id": "app_id_value",
    "biz_type": "biz_type_value",
    "biz_unique_id": "biz_unique_id_value",
    "recognition_mode": "default"
  }
}
```

#### 3.6.2 根据应用ID查询任务列表

**接口地址**：`GET /list_tasks_by_app`

**请求参数**：

| 参数名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| app_id | string | 否 | 应用ID |
| biz_type | string | 否 | 业务类型 |
| recognition_mode | string | 否 | 识别模式 |
| page | int | 否 | 页码，默认1 |
| page_size | int | 否 | 每页数量，默认10，最大100 |

**返回结果**：

```json
{
  "code": 0,
  "msg": "查询任务列表成功",
  "tasks": [
    {
      "task_id": "task_id_value",
      "task_type": "file_upload",
      "file_name": "filename.wav",
      "status": "completed",
      "created_time": 1620000000,
      "updated_time": 1620000000,
      "callback_status": "success",
      "biz_type": "biz_type_value",
      "biz_unique_id": "biz_unique_id_value",
      "recognition_mode": "meeting"
    },
    ...
  ],
  "total": 100,
  "page": 1,
  "limit": 10
}
```

## 4. 任务状态说明

| 状态值 | 说明 |
|--------|------|
| pending | 任务待处理 |
| processing | 任务处理中 |
| completed | 任务处理完成 |
| failed | 任务处理失败 |
| canceled | 任务已取消 |

## 5. 错误码说明

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1 | 任务尚未完成 |
| 2 | 任务处理失败 |
| 3 | 任务已取消 |
| 4 | 结果解析失败 |

## 6. 使用示例

### 6.1 标准识别模式示例

**单任务提交（标准模式）：**
```bash
curl -X POST "http://localhost:8000/submit_task" \
  -F "file=@test.wav" \
  -F "file_name=test.wav" \
  -F "recognition_mode=default"
```

**批量任务提交（标准模式）：**
```bash
curl -X POST "http://localhost:8000/batch_operation" \
  -F "operation=submit" \
  -F "file_list=[{\"file_url\":\"http://example.com/audio1.wav\",\"file_name\":\"audio1.wav\"},{\"file_url\":\"http://example.com/audio2.wav\",\"file_name\":\"audio2.wav\"}]" \
  -F "recognition_mode=default"
```

### 6.2 会议识别模式示例

**单任务提交（会议模式）：**
```bash
curl -X POST "http://localhost:8000/submit_task" \
  -F "file=@meeting.wav" \
  -F "file_name=meeting.wav" \
  -F "recognition_mode=meeting"
```

**批量任务提交（会议模式）：**
```bash
curl -X POST "http://localhost:8000/batch_operation" \
  -F "operation=submit" \
  -F "file_list=[{\"file_url\":\"http://example.com/meeting1.wav\",\"file_name\":\"meeting1.wav\"}]" \
  -F "recognition_mode=meeting"
```

### 6.3 查询识别模式任务

**根据识别模式查询任务列表：**
```bash
curl -X GET "http://localhost:8000/list_tasks_by_app?recognition_mode=meeting&page=1&page_size=10"
```

### 6.4 典型使用场景

**场景1：播客音频处理**
- 音频特点：多人对话，有主持人、嘉宾
- 推荐模式：`meeting`
- 理由：能够区分不同说话人，生成对话格式结果

**场景2：会议录音处理**
- 音频特点：多人会议，需要知道谁在什么时候说了什么
- 推荐模式：`meeting`
- 理由：说话人识别功能，能够生成会议记录格式

**场景3：个人语音笔记**
- 音频特点：单人录制，主要是备忘信息
- 推荐模式：`default`
- 理由：只需要文字转换，不需要说话人信息

**场景4：演讲录音**
- 音频特点：单人演讲，内容结构清晰
- 推荐模式：`default`
- 理由：专注于内容识别，效率更高

**场景5：客服通话录音**
- 音频特点：客服和客户对话
- 推荐模式：`meeting`
- 理由：需要区分客服和客户的话术，便于分析

## 7. 性能建议

### 7.1 识别模式选择建议

| 音频特征 | 推荐模式 | 性能考虑 |
|----------|----------|----------|
| 单人语音 | default | 速度更快，资源消耗少 |
| 多人对话 | meeting | 准确识别说话人，但需要更多计算资源 |
| 背景噪音大 | default | 专注于语音识别，减少说话人识别干扰 |
| 音质清晰 | meeting | 充分利用说话人识别功能 |

### 7.2 批量处理建议

1. **合理设置并发数**：根据服务器配置调整 `max_concurrent_tasks`
2. **使用批量接口**：批量操作比单次操作效率更高
3. **文件大小控制**：建议单个音频文件不超过100MB
4. **格式标准化**：使用WAV格式可获得最佳识别效果

### 7.3 监控和调试

1. **查看任务状态**：使用 `/get_task_status` 监控处理进度
2. **批量查询结果**：使用 `/batch_get_task_result` 批量获取结果
3. **日志监控**：关注服务器日志，了解处理状态和错误信息
4. **回调机制**：配置回调URL接收异步处理结果 |