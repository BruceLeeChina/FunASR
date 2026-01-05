# TTS语音合成API接口文档

## 概述

本API提供文本转语音（TTS）合成服务，支持并发处理和HLS流媒体播放。服务基于Flask框架构建，支持多路音频流同时处理。

## 基础信息

- **基础URL**: `http://{host}:5000`
- **默认端口**: 5000
- **HLS服务端口**: 9080
- **RTSP服务端口**: 8554
- **响应格式**: JSON
- **字符编码**: UTF-8

---

## 典型使用流程

### 典型调用流程

1. **发起合成请求**
   ```bash
   POST /synthesize-and-push
   {
       "text": "需要合成的文本内容"
   }
   ```

2. **轮询状态查询**
   ```bash
   GET /stream-status/{req_id}
   ```
   轮询直到`status`变为`"ready"`（建议轮询间隔5秒，最大尝试次数60次）

3. **获取结果**
   - 方式1：使用HLS流播放（推荐）
     ```
     HLS URL: http://{host}:9080/hls/test_{req_id}/index.m3u8
     ```
   - 方式2：下载WAV文件
     ```
     GET /download-wav/{req_id}
     ```
   - 方式3：列出并访问音频文件
     ```
     GET /audio/{req_id}/
     GET /audio/{req_id}/{filename}
     ```

4. **清理资源**
   ```bash
   POST /stop-stream/{req_id}
   ```
   

## 接口列表

### 1. 健康检查接口

检查服务运行状态和TTS模型初始化情况。

#### 接口信息
- **URL**: `/health`
- **方法**: GET

#### 请求示例
```
GET /health
```

#### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `status` | string | 服务状态（"healthy"表示正常） |
| `service` | string | 服务名称 |
| `timestamp` | float | 时间戳（Unix时间戳，秒） |
| `tts_initialized` | boolean | TTS模型是否已初始化 |

#### 响应示例
```json
{
    "status": "healthy",
    "service": "TTS-RTSP-HLS",
    "timestamp": 1733988612.345,
    "tts_initialized": true
}
```

#### 状态码
- `200`: 服务正常
- `500`: 服务异常

---

### 2. TTS模型测试接口

测试TTS模型是否正常工作，生成测试音频。

#### 接口信息
- **URL**: `/test-tts`
- **方法**: GET

#### 请求示例
```
GET /test-tts
```

#### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `status` | string | 测试状态（"success"表示成功） |
| `message` | string | 测试结果描述 |
| `file_size` | integer | 生成的测试音频文件大小（字节） |

#### 响应示例
```json
{
    "status": "success",
    "message": "TTS模型工作正常",
    "file_size": 12345
}
```

#### 状态码
- `200`: 测试成功
- `500`: 测试失败（模型初始化异常）

---

### 3. 语音合成接口

启动新的TTS合成任务，生成音频并转换为HLS流。

#### 接口信息
- **URL**: `/synthesize-and-push`
- **方法**: POST
- **Content-Type**: `application/json`

#### 请求参数

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `text` | string | 是 | 需要合成的文本内容（支持中文） |

#### 请求示例
```json
{
    "text": "你好，这是一个测试文本。"
}
```

#### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `req_id` | string | 请求唯一标识符（8位UUID） |
| `rtsp_url` | string | RTSP流地址（格式：rtsp://{host}:8554/test_{req_id}） |
| `hls_url` | string | HLS流地址（格式：http://{host}:9080/hls/test_{req_id}/index.m3u8） |
| `status` | string | 当前状态（"processing"表示处理中） |
| `message` | string | 状态描述信息 |

#### 响应示例
```json
{
    "req_id": "a1b2c3d4",
    "rtsp_url": "rtsp://192.168.21.164:8554/test_a1b2c3d4",
    "hls_url": "http://192.168.21.164:9080/hls/test_a1b2c3d4/index.m3u8",
    "status": "processing",
    "message": "合成和HLS转换已开始，请稍后..."
}
```

#### 状态码
- `200`: 请求已接受，开始处理
- `400`: 请求参数错误（文本为空）
- `500`: 服务器内部错误

---

### 4. 状态查询接口

查询指定合成请求的状态信息。

#### 接口信息
- **URL**: `/stream-status/<req_id>`
- **方法**: GET
- **路径参数**: `req_id` - 请求ID

#### 请求示例
```
GET /stream-status/a1b2c3d4
```

#### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `req_id` | string | 请求ID |
| `status` | string | 状态：`pending`(待处理)、`processing`(处理中)、`ready`(就绪)、`failed`(失败) |
| `message` | string | 状态详细信息 |
| `hls_exists` | boolean | HLS文件是否存在 |
| `hls_running` | boolean | HLS转换进程是否在运行 |
| `rtsp_url` | string | RTSP流地址 |
| `hls_url` | string | HLS流地址 |
| `hls_file_size` | integer | HLS文件大小（字节） |
| `hls_segment_count` | integer | HLS片段数量 |
| `estimated_duration` | float | 估计音频时长（秒） |
| `wav_direct_url` | string | WAV文件直接下载URL（相对于API服务器的路径） |
| `wav_file_exists` | boolean | WAV文件是否存在 |

#### 响应示例（处理中）
```json
{
    "req_id": "a1b2c3d4",
    "status": "processing",
    "message": "TTS合成完成，开始HLS转换...",
    "hls_exists": false,
    "hls_running": true,
    "rtsp_url": "rtsp://192.168.21.164:8554/test_a1b2c3d4",
    "hls_url": "http://192.168.21.164:9080/hls/test_a1b2c3d4/index.m3u8",
    "wav_file_exists": true
}
```

#### 响应示例（就绪）
```json
{
    "req_id": "a1b2c3d4",
    "status": "ready",
    "message": "HLS转换成功，时长: 12.5秒",
    "hls_exists": true,
    "hls_running": false,
    "rtsp_url": "rtsp://192.168.21.164:8554/test_a1b2c3d4",
    "hls_url": "http://192.168.21.164:9080/hls/test_a1b2c3d4/index.m3u8",
    "hls_file_size": 12345,
    "hls_segment_count": 7,
    "estimated_duration": 12.5,
    "wav_direct_url": "/audio/a1b2c3d4/tts_output_1234567890.wav",
    "wav_file_exists": true
}
```

#### 状态码
- `200`: 查询成功
- `404`: 请求ID不存在

---

### 5. 批量状态查询接口

查询所有活动流的状态信息。

#### 接口信息
- **URL**: `/stream-status`
- **方法**: GET

#### 请求示例
```
GET /stream-status
```

#### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `active_streams` | integer | 活动流数量 |
| `streams` | array | 流状态列表 |

#### 流状态对象结构

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `req_id` | string | 请求ID |
| `status` | string | 当前状态 |
| `message` | string | 状态描述 |
| `rtsp_url` | string | RTSP流地址 |
| `hls_url` | string | HLS流地址 |

#### 响应示例
```json
{
    "active_streams": 2,
    "streams": [
        {
            "req_id": "a1b2c3d4",
            "status": "ready",
            "message": "HLS转换成功，时长: 12.5秒",
            "rtsp_url": "rtsp://192.168.21.164:8554/test_a1b2c3d4",
            "hls_url": "http://192.168.21.164:9080/hls/test_a1b2c3d4/index.m3u8"
        },
        {
            "req_id": "e5f6g7h8",
            "status": "processing",
            "message": "开始TTS合成...",
            "rtsp_url": "rtsp://192.168.21.164:8554/test_e5f6g7h8",
            "hls_url": "http://192.168.21.164:9080/hls/test_e5f6g7h8/index.m3u8"
        }
    ]
}
```

---

### 6. 音频文件访问接口

#### 6.1 下载WAV文件

下载合成的WAV音频文件。

##### 接口信息
- **URL**: `/download-wav/<req_id>`
- **方法**: GET
- **路径参数**: `req_id` - 请求ID

##### 请求示例
```
GET /download-wav/a1b2c3d4
```

##### 响应
- **Content-Type**: `audio/wav`
- **Content-Disposition**: `attachment; filename="tts_output_a1b2c3d4_20241212_143022.wav"`

##### 状态码
- `200`: 下载成功
- `404`: 文件不存在或尚未生成
- `500`: 服务器内部错误

---

#### 6.2 列出音频文件

列出指定请求的所有音频文件。

##### 接口信息
- **URL**: `/audio/<req_id>/`
- **方法**: GET
- **路径参数**: `req_id` - 请求ID

##### 请求示例
```
GET /audio/a1b2c3d4/
```

##### 响应参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `req_id` | string | 请求ID |
| `files` | array | 文件列表 |
| `count` | integer | 文件数量 |

##### 文件对象结构

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `name` | string | 文件名 |
| `size` | integer | 文件大小（字节） |
| `url` | string | 文件访问URL（相对于API服务器的路径） |
| `modified` | string | 修改时间（格式：YYYY-MM-DD HH:MM:SS） |

##### 响应示例
```json
{
    "req_id": "a1b2c3d4",
    "files": [
        {
            "name": "tts_output_1234567890.wav",
            "size": 123456,
            "url": "/audio/a1b2c3d4/tts_output_1234567890.wav",
            "modified": "2024-12-12 14:30:22"
        }
    ],
    "count": 1
}
```

---

#### 6.3 直接访问WAV文件

直接获取WAV文件，适用于内嵌播放。

##### 接口信息
- **URL**: `/audio/<req_id>/<filename>`
- **方法**: GET
- **路径参数**: 
  - `req_id` - 请求ID
  - `filename` - 文件名（从文件列表接口获取）

##### 请求示例
```
GET /audio/a1b2c3d4/tts_output_1234567890.wav
```

##### 响应
- **Content-Type**: `audio/wav`
- **Cache-Control**: `public, max-age=86400`
- **Access-Control-Allow-Origin**: `*`

##### 状态码
- `200`: 文件获取成功
- `403`: 访问被拒绝（路径安全检查失败）
- `404`: 文件不存在

---

### 7. HLS流访问接口

#### 7.1 获取HLS播放列表

获取用于播放的HLS m3u8文件。

##### 接口信息
- **URL**: `/hls/<hls_app_subdir>/index.m3u8`
- **方法**: GET
- **路径参数**: `hls_app_subdir` - HLS应用子目录（格式：`test_<req_id>`）

##### 请求示例
```
GET /hls/test_a1b2c3d4/index.m3u8
```

##### 响应
- **Content-Type**: `application/vnd.apple.mpegurl`
- **Cache-Control**: `no-cache, no-store, must-revalidate`

##### 状态码
- `200`: 获取成功
- `403`: 访问被拒绝（路径安全检查失败）
- `404`: 文件不存在

---

#### 7.2 获取HLS片段文件

获取HLS的TS片段文件。

##### 接口信息
- **URL**: `/hls/<hls_app_subdir>/<segment_file>`
- **方法**: GET
- **路径参数**: 
  - `hls_app_subdir` - HLS应用子目录
  - `segment_file` - 片段文件名（如：`segment_001.ts`）

##### 请求示例
```
GET /hls/test_a1b2c3d4/segment_001.ts
```

##### 响应
- **Content-Type**: `video/mp2t`
- **Cache-Control**: `no-cache, no-store, must-revalidate`

---

### 8. 控制接口

#### 8.1 停止单个流

停止指定请求的合成和流处理。

##### 接口信息
- **URL**: `/stop-stream/<req_id>`
- **方法**: POST
- **路径参数**: `req_id` - 请求ID

##### 请求示例
```
POST /stop-stream/a1b2c3d4
```

##### 响应示例
```json
{
    "message": "请求 a1b2c3d4 的流已停止"
}
```

##### 状态码
- `200`: 停止成功
- `404`: 请求不存在
- `500`: 停止过程出错

---

#### 8.2 停止所有流

停止所有活动的合成和流处理。

##### 接口信息
- **URL**: `/stop-stream`
- **方法**: POST

##### 请求示例
```
POST /stop-stream
```

##### 响应示例
```json
{
    "message": "已停止所有推流与HLS转换"
}
```

=========================================================== END ====================================================================
---

## API测试工具

### 测试脚本使用方法

```bash
# 安装依赖
pip install requests pydub

# 运行完整测试（默认参数）
python tts_api_tester.py

# 自定义API地址和文本
python tts_api_tester.py --url http://192.168.1.100:5000 --text "自定义测试文本"

# 只进行健康检查
python tts_api_tester.py --skip-full

# 指定HLS服务地址
python tts_api_tester.py --url http://localhost:5000 --hls-url http://localhost:9080
```

### 测试流程说明

测试脚本按照以下流程执行：
1. 健康检查 (`/health`)
2. TTS模型测试 (`/test-tts`)
3. 语音合成请求 (`/synthesize-and-push`)
4. 轮询流状态 (`/stream-status/<req_id>`)
5. HLS访问测试 (直接访问HLS URL)
6. WAV文件下载测试 (`/download-wav/<req_id>`)
7. 直接音频访问测试 (`/audio/<req_id>/`)
8. 批量状态查询 (`/stream-status`)
9. 停止单个流测试 (`/stop-stream/<req_id>`)
10. 停止所有流测试 (`/stop-stream`)

### 测试结果

测试完成后会生成详细日志文件：
- 文件名格式：`tts_api_test_YYYYMMDD_HHMMSS.log`
- 包含所有请求响应和验证结果

---


### 前端集成示例

```javascript
// 1. 发起合成请求
async function synthesizeText(text) {
    const response = await fetch('/synthesize-and-push', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    return await response.json();
}

// 2. 轮询状态
async function pollStatus(reqId, maxAttempts = 60, interval = 5000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const response = await fetch(`/stream-status/${reqId}`);
        const status = await response.json();
        
        console.log(`轮询尝试 ${attempt}/${maxAttempts}: ${status.status}`);
        
        if (status.status === 'ready') {
            // 3. 使用HLS播放
            const audioElement = document.getElementById('audio-player');
            if (Hls.isSupported()) {
                const hls = new Hls();
                hls.loadSource(status.hls_url);
                hls.attachMedia(audioElement);
            } else if (audioElement.canPlayType('application/vnd.apple.mpegurl')) {
                audioElement.src = status.hls_url;
            }
            
            return status;
        } else if (status.status === 'failed') {
            throw new Error(`合成失败: ${status.message}`);
        }
        
        // 继续轮询
        await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    throw new Error('轮询超时');
}

// 4. 清理资源
async function stopStream(reqId) {
    const response = await fetch(`/stop-stream/${reqId}`, {
        method: 'POST'
    });
    return await response.json();
}

// 使用示例
async function runTTS() {
    try {
        const text = document.getElementById('text-input').value;
        const result = await synthesizeText(text);
        console.log(`请求ID: ${result.req_id}`);
        
        const status = await pollStatus(result.req_id);
        console.log('合成完成:', status.message);
        
        // 5分钟后自动清理
        setTimeout(() => {
            stopStream(result.req_id);
            console.log('资源已清理');
        }, 300000);
        
    } catch (error) {
        console.error('TTS处理失败:', error);
    }
}
```

## 注意事项

1. **异步处理**：合成请求是异步的，需要轮询状态接口获取进度
2. **轮询建议**：
   - 初始等待：发起请求后等待60秒再进行第一次轮询
   - 轮询间隔：5秒
   - 最大尝试：60次（总时长约5分钟）
3. **资源清理**：建议处理完成后调用停止接口释放资源
4. **并发限制**：系统支持多路并发，但实际并发数受服务器资源限制
5. **自动清理**：系统会自动清理超过2小时的旧请求资源
6. **HLS兼容性**：HLS流需要客户端支持HLS协议（现代浏览器或播放器）
7. **音频格式**：输出为WAV格式，采样率44100Hz，立体声，128kbps码率

## 错误码说明

| 状态码 | 含义 | 解决方法 |
|--------|------|----------|
| 400 | 请求参数错误 | 检查请求体格式和必填参数 |
| 403 | 访问被拒绝 | 检查文件路径安全性，确保不包含非法字符 |
| 404 | 资源不存在 | 检查请求ID是否正确，资源可能已被清理 |
| 500 | 服务器内部错误 | 检查服务器日志，联系管理员 |

### 常见问题处理

1. **轮询超时**：检查TTS模型是否初始化，服务器资源是否充足
2. **HLS访问失败**：确保HLS服务端口(9080)可访问，防火墙设置正确
3. **音频文件不存在**：等待合成完成，确保状态为"ready"后再下载
4. **RTSP连接失败**：检查RTSP服务端口(8554)和客户端播放器支持

## 性能指标

- **单次请求处理时间**: 文本长度决定，平均约2-10秒
- **最大文本长度**: 无硬性限制，但长文本会分段处理
- **并发能力**: 受服务器CPU和内存限制，建议不超过10路并发
- **输出质量**: 16-bit PCM WAV，44100Hz采样率，立体声
- **流媒体延迟**: HLS延迟约2-5秒
- **文件大小**: 每秒音频约176KB（44.1kHz × 16-bit × 2声道）

## 测试建议

1. **基础测试**：使用测试脚本验证所有接口功能
2. **压力测试**：逐步增加并发请求，观察系统表现
3. **兼容性测试**：在不同浏览器/播放器测试HLS流播放
4. **网络测试**：在不同网络环境下测试下载和流媒体播放

---

**版本**: 2.0  
**最后更新**: 2024-12-12  
**维护者**: TTS-RTSP-HLS开发团队  
**测试工具**: tts_api_tester.py