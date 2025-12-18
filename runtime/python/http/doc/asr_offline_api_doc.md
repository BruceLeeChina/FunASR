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
python server.py --host 0.0.0.0 --port 8000 --device cpu --ngpu 0
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

## 3. API接口列表

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
  "callback_status": "pending"
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
  "callback_status": "success"
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
    "callback_status": "success"
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
      "callback_status": "success"
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
      "callback_status": "success"
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
      "callback_status": "success"
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
    "biz_unique_id": "biz_unique_id_value"
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
      "biz_unique_id": "biz_unique_id_value"
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