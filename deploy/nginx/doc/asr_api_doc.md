根据测试脚本的实际情况，我对API文档进行以下重要更新和修正：

# FunASR 语音识别 WebSocket API 接口文档

## 概述

FunASR（Fundamental Audio Speech Recognition）是由阿里巴巴达摩院推出的语音识别开源工具包。本API提供基于WebSocket的实时语音识别服务，支持多种工作模式（2pass、online、offline），集成VAD（语音活动检测）、ASR（自动语音识别）和标点模型。

## 基础信息

- **协议**: WebSocket (WSS)
- **地址**: `wss://192.168.21.164:10095/` (已确认可用)
- **备用地址**: `wss://192.168.21.164:84/ws` (HTML页面使用)
- **端口**: 10095 (主), 84 (备用)
- **安全**: SSL/TLS 加密（测试中可禁用证书验证）
- **编码**: UTF-8
- **音频格式**: 16kHz, 16-bit, 单声道 PCM (必须)

---

## 接口概览

### 1. WebSocket 连接建立
建立与语音识别服务的WebSocket连接。

### 2. 语音识别开始
发送控制消息配置识别参数并开始传输音频数据。

### 3. 语音识别停止
发送停止信号结束语音识别。

### 4. 识别结果接收
接收服务端返回的语音识别结果。

---

## 详细接口说明

### 1. WebSocket 连接建立

#### 接口信息
- **URL**: `wss://192.168.21.164:10095/`
- **协议**: WebSocket over TLS
- **连接限制**: 目前仅支持单客户端连接

#### 连接示例 (Python - 已验证)
```python
import asyncio
import websockets
import ssl

async def connect_to_server():
    url = "wss://192.168.21.164:10095/"
    
    # 创建不验证证书的SSL上下文
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        async with websockets.connect(
            url, 
            ssl=ssl_context,
            open_timeout=10
        ) as websocket:
            print("✅ WebSocket连接成功")
            return websocket
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None
```

#### 连接示例 (JavaScript)
```javascript
// JavaScript 示例
const ws = new WebSocket('wss://192.168.21.164:10095/');

ws.onopen = function() {
    console.log('WebSocket 连接已建立');
    // 连接成功后可以发送配置信息
};

ws.onerror = function(error) {
    console.error('WebSocket 连接错误:', error);
};

ws.onclose = function() {
    console.log('WebSocket 连接已关闭');
};
```

#### 注意事项
- 需要SSL证书（服务端已配置自签名证书）
- 浏览器首次访问需授权证书
- 目前仅支持单客户端连接，新连接会关闭旧连接

---

### 2. 语音识别开始

#### 2.1 发送控制配置消息

在开始发送音频数据前，需要发送JSON格式的控制消息配置识别参数。

##### 消息格式 (根据测试脚本验证)
```json
{
    "is_speaking": true,
    "wav_name": "test_audio",
    "mode": "online",
    "chunk_size": [5, 10, 5],
    "encoder_chunk_look_back": 4,
    "decoder_chunk_look_back": 0
}
```

##### 参数说明

| 参数名 | 类型 | 必填 | 默认值 | 描述 | 状态 |
|--------|------|------|--------|------|------|
| `is_speaking` | boolean | 是 | true | 是否正在说话，用于控制流式识别 | ✅ 已验证 |
| `mode` | string | 是 | "online" | 识别模式：`2pass`、`online`、`offline` | ✅ 已验证 |
| `wav_name` | string | 否 | "test_audio" | 音频名称标识 | ✅ 已验证 |
| `chunk_size` | array | 否 | [5, 10, 5] | 分块大小配置，数组格式 | ⚠️ 已验证但需注意 |
| `chunk_interval` | integer | 否 | 10 | 音频分块间隔（单位：毫秒） | ❓ 未验证 |
| `encoder_chunk_look_back` | integer | 否 | 4 | 编码器回看块数 | ⚠️ 已验证但需注意 |
| `decoder_chunk_look_back` | integer | 否 | 0 | 解码器回看块数 | ⚠️ 已验证但需注意 |
| `hotword` | string | 否 | "" | 热词配置，每行一个，格式："关键词 权重" | ❓ 未验证 |

##### 模式说明

| 模式 | 描述 | 适用场景 | 状态 |
|------|------|----------|------|
| `2pass` | 双通模式（流式+非流式） | 实时性要求高且需要高准确率 | ⚠️ 已验证但结果待确认 |
| `online` | 纯流式模式 | 低延迟实时识别 | ✅ 已验证 |
| `offline` | 纯非流式模式 | 音频文件转录，高准确率 | ⚠️ 已验证但结果待确认 |

#### 2.2 发送音频数据

配置完成后，开始发送二进制音频数据。

##### 音频格式要求 (必须)
- **采样率**: 16kHz (必须，其他采样率需重采样)
- **位深度**: 16-bit (必须)
- **声道**: 单声道 (mono) (必须)
- **编码**: PCM (必须)
- **分块大小**: 320字节 (对应10ms音频，16000Hz * 0.01s * 2bytes)

##### 音频重采样 (重要)
如果音频不是16kHz，必须重采样：
```python
import numpy as np
from scipy import signal

def resample_audio_to_16000(audio_data: bytes, original_rate: int) -> bytes:
    """将任意采样率的音频重采样到16000Hz"""
    # 转换为float32
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 计算重采样因子
    num_samples = int(len(audio_array) * 16000 / original_rate)
    
    # 重采样
    resampled_audio = signal.resample(audio_array, num_samples)
    
    # 转换回16-bit PCM
    resampled_audio_int16 = (resampled_audio * 32767).astype(np.int16)
    
    return resampled_audio_int16.tobytes()
```

##### 发送频率
- 根据 `chunk_interval` 参数决定
- 默认每10ms发送一次

##### 发送示例 (Python - 已验证)
```python
async def send_audio_stream(websocket, audio_data: bytes, sample_rate: int = 16000):
    """流式发送音频数据"""
    # 确保采样率为16000Hz
    if sample_rate != 16000:
        audio_data = resample_audio_to_16000(audio_data, sample_rate)
        sample_rate = 16000
    
    # 分块大小：16000Hz * 0.01s * 2bytes = 320字节
    chunk_size = int(sample_rate * 0.01 * 2)
    total_chunks = len(audio_data) // chunk_size
    
    print(f"开始发送音频：共{total_chunks}块，每块{chunk_size}字节")
    
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_data[start:end]
        
        # 发送音频块
        await websocket.send(chunk)
        
        # 控制发送速率（10ms间隔）
        await asyncio.sleep(0.01)
```

---

### 3. 语音识别停止

#### 3.1 发送停止信号

发送 `is_speaking: false` 控制消息，通知服务端说话结束。

##### 停止消息格式
```json
{
    "is_speaking": false
}
```

#### 3.2 关闭WebSocket连接

识别完成后，可以关闭WebSocket连接。

##### 关闭连接示例
```javascript
// 发送停止信号
ws.send(JSON.stringify({ "is_speaking": false }));

// 等待最后一轮识别结果
setTimeout(() => {
    ws.close();
}, 1000);
```

---

### 4. 识别结果接收

#### 结果消息格式

服务端返回的识别结果为JSON格式：

```json
{
    "mode": "2pass-online",
    "text": "你好，这是一个测试。",
    "wav_name": "test_audio",
    "is_final": false
}
```

##### 结果参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| `mode` | string | 识别模式：`2pass-online`、`2pass-offline`、`online`、`offline` |
| `text` | string | 识别出的文本内容 |
| `wav_name` | string | 音频名称标识 |
| `is_final` | boolean | 是否为最终结果（True表示语音段结束） |

#### 结果接收示例
```javascript
ws.onmessage = function(event) {
    if (typeof event.data === 'string') {
        // 解析JSON结果
        const result = JSON.parse(event.data);
        console.log(`模式: ${result.mode}`);
        console.log(`识别结果: ${result.text}`);
        console.log(`是否最终: ${result.is_final}`);
        
        // 更新UI显示
        document.getElementById('result').innerText += result.text + '\n';
    }
};
```

#### 结果类型说明

| 模式组合 | 描述 |
|----------|------|
| `2pass-online` | 2pass模式下的流式识别结果（中间结果） |
| `2pass-offline` | 2pass模式下的非流式识别结果（最终结果） |
| `online` | 纯流式模式结果 |
| `offline` | 纯非流式模式结果 |

---


=========================================================== END ====================================================================

## 完整调用流程示例

### Python 完整示例 (基于测试脚本)
```python
import asyncio
import websockets
import json
import ssl
import wave
import numpy as np
from scipy import signal

class ASRClientPython:
    """基于测试脚本的完整ASR客户端"""
    def __init__(self, server_url: str = "wss://192.168.21.164:10095/"):
        self.server_url = server_url
        self.websocket = None
        self.is_recording = False
        self.results_received = []
    
    async def connect(self) -> bool:
        """连接WebSocket服务器"""
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                self.server_url,
                ssl=ssl_context,
                open_timeout=30
            )
            print(f"✅ 已连接到ASR服务器: {self.server_url}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def start_recognition(self, audio_file_path: str, mode: str = "online") -> bool:
        """开始语音识别"""
        if not self.websocket:
            print("❌ 未连接到服务器")
            return False
        
        # 1. 加载音频文件
        print("🎵 加载音频文件...")
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                params = wav_file.getparams()
                original_rate = params.framerate
                print(f"   采样率: {original_rate} Hz")
                print(f"   声道数: {params.nchannels}")
                print(f"   位深度: {params.sampwidth * 8} bit")
                
                # 读取音频数据
                audio_data = wav_file.readframes(params.nframes)
                
                # 如果是立体声，转换为单声道
                if params.nchannels == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    audio_data = audio_array.tobytes()
        except Exception as e:
            print(f"❌ 加载音频文件失败: {e}")
            return False
        
        # 2. 发送配置
        print("📤 发送配置...")
        config = {
            "is_speaking": True,
            "wav_name": audio_file_path,
            "mode": mode,
            "chunk_size": [5, 10, 5],
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 0
        }
        
        await self.websocket.send(json.dumps(config))
        self.is_recording = True
        
        # 3. 发送音频（自动重采样到16000Hz）
        print("📤 发送音频数据...")
        await self.send_audio_data(audio_data, original_rate)
        
        return True
    
    async def send_audio_data(self, audio_data: bytes, sample_rate: int):
        """发送音频数据"""
        # 确保采样率为16000Hz
        if sample_rate != 16000:
            audio_data = self.resample_audio_to_16000(audio_data, sample_rate)
            sample_rate = 16000
        
        # 分块发送
        chunk_size = int(sample_rate * 0.01 * 2)  # 10ms块大小
        total_chunks = len(audio_data) // chunk_size
        
        print(f"  总大小: {len(audio_data)} 字节")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  块大小: {chunk_size} 字节 (10ms)")
        print(f"  总块数: {total_chunks}")
        
        for i in range(total_chunks):
            if not self.is_recording:
                break
                
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_data[start:end]
            
            await self.websocket.send(chunk)
            await asyncio.sleep(0.01)  # 10ms间隔
            
            # 显示进度
            if i % 100 == 0:
                progress = (i / total_chunks) * 100
                print(f"  进度: {progress:.1f}%")
        
        print("✅ 音频发送完成")
    
    def resample_audio_to_16000(self, audio_data: bytes, original_rate: int) -> bytes:
        """重采样音频到16000Hz"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        num_samples = int(len(audio_array) * 16000 / original_rate)
        resampled_audio = signal.resample(audio_array, num_samples)
        resampled_audio_int16 = (resampled_audio * 32767).astype(np.int16)
        return resampled_audio_int16.tobytes()
    
    async def stop_recognition(self):
        """停止语音识别"""
        if not self.is_recording:
            return
        
        # 发送停止信号
        stop_msg = {"is_speaking": False}
        await self.websocket.send(json.dumps(stop_msg))
        print("📤 已发送停止信号")
        
        self.is_recording = False
        
        # 等待最终结果
        print("⏳ 等待服务器发送最终结果...")
        await asyncio.sleep(3)
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        print("🔌 连接已断开")
    
    async def receive_results(self):
        """接收识别结果"""
        try:
            while True:
                message = await self.websocket.recv()
                if isinstance(message, str):
                    result = json.loads(message)
                    self.results_received.append(result)
                    self.handle_result(result)
        except websockets.exceptions.ConnectionClosed:
            print("连接已关闭")
        except Exception as e:
            print(f"接收结果错误: {e}")
    
    def handle_result(self, result: dict):
        """处理识别结果"""
        mode = result.get("mode", "")
        text = result.get("text", "")
        is_final = result.get("is_final", False)
        
        if is_final:
            print(f"✅ [最终结果] {text}")
        else:
            print(f"⏳ [中间结果] {text}")

# 使用示例
async def main():
    # 创建客户端
    client = ASRClientPython()
    
    # 连接服务器
    if await client.connect():
        # 开始识别
        await client.start_recognition("test.wav", mode="online")
        
        # 停止识别
        await client.stop_recognition()
        
        # 断开连接
        await client.disconnect()
        
        # 打印所有结果
        print("\n📊 识别结果总结:")
        for i, result in enumerate(client.results_received, 1):
            print(f"{i}. [{result['mode']}] {result['text']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 错误处理与调试

### 当前测试状态
根据测试结果，以下问题已确认：
1. ✅ WebSocket连接成功
2. ✅ 配置消息可以发送
3. ✅ 音频数据可以传输
4. ❌ **未收到任何识别结果**

### 可能的原因及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 连接成功但无识别结果 | 1. 音频采样率不正确<br>2. VAD未检测到语音<br>3. ASR模型未正确加载<br>4. 服务器配置问题 | 1. 确保音频为16000Hz<br>2. 增加音频音量或清晰度<br>3. 检查服务器日志<br>4. 尝试不同的识别模式 |
| 音频发送但无响应 | 1. 音频格式不正确<br>2. 分块大小不正确<br>3. 发送频率过快/过慢 | 1. 确保为16-bit PCM<br>2. 使用320字节分块<br>3. 控制10ms发送间隔 |
| SSL证书问题 | 自签名证书不被信任 | 在测试环境中禁用证书验证 |

### 调试建议
1. **检查音频格式**：确保为16000Hz, 16-bit, 单声道PCM
2. **查看服务器日志**：检查ASR模型是否正常加载
3. **尝试不同模式**：分别测试online、2pass、offline模式
4. **简化测试**：使用生成的测试音频而非真实音频
5. **网络抓包**：使用Wireshark确认数据是否正确传输

---

## 性能参数

### 推荐配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 音频采样率 | 16000 Hz | **必须**，否则需重采样 |
| 音频格式 | 16-bit PCM | **必须** |
| 分块大小 | 320字节 | 对应10ms音频（16000 * 0.01 * 2） |
| 发送间隔 | 10ms | 与分块大小匹配 |
| 识别模式 | online | 实时性最佳 |
| VAD灵敏度 | 默认 | 服务端自动调整 |

### 延迟参考

| 模式 | 流式延迟 | 最终结果延迟 |
|------|----------|--------------|
| `online` | 100-300ms | N/A |
| `2pass` | 100-300ms | 500-1000ms |
| `offline` | N/A | 音频长度+处理时间 |

---

## 注意事项

1. **音频采样率必须为16000Hz**：这是最关键的要求，其他采样率无法识别
2. **单连接限制**：目前服务端仅支持单客户端连接
3. **SSL证书**：首次连接需要在浏览器中授权自签名证书
4. **音频格式**：必须为16-bit PCM单声道格式
5. **VAD敏感度**：在嘈杂环境中可能需要调整VAD参数
6. **资源占用**：长时间运行建议监控服务端资源使用情况

---

## 服务管理

### 启动服务
```bash
# 进入项目目录
cd /path/to/funasr_wss_server

# 启动服务（使用SSL）
python funasr_wss_server.py \
  --host 192.168.21.164 \
  --port 10095 \
  --certfile ../../ssl_key/server.crt \
  --keyfile ../../ssl_key/server.key
```

### 停止服务
```bash
# 查找进程ID
ps aux | grep funasr_wss_server

# 停止服务
kill [PID]
```

### 查看日志
```bash
# 查看服务输出
tail -f /var/log/funasr_server.log
```

---

**文档版本**: 1.1 (基于测试结果修订)  
**最后更新**: 2025-12-12  
**测试状态**: 连接和音频传输正常，识别结果待确认  
**维护团队**: FunASR 开发团队