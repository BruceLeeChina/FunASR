# pip install numpy scipy websockets
import asyncio
import websockets
import json
import ssl
import certifi
from typing import Dict, Optional, List
import wave
import struct
import os
import time
import sys
import numpy as np

try:
    from scipy import signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  未安装scipy，将无法进行重采样")


class ASRClientPython:
    def __init__(self, server_url: str = "wss://192.168.21.130:10095/"):
        self.server_url = server_url
        self.websocket = None
        self.is_recording = False
        self.results_received = []
        self.debug_mode = True

    def create_ssl_context(self, verify_cert: bool = False) -> Optional[ssl.SSLContext]:
        """
        创建SSL上下文
        """
        if self.server_url.startswith('wss://'):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

            if verify_cert:
                ssl_context.load_verify_locations(certifi.where())
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            return ssl_context
        return None

    async def connect(self, verify_cert: bool = False) -> bool:
        """连接WebSocket服务器"""
        try:
            ssl_context = self.create_ssl_context(verify_cert)

            self.websocket = await websockets.connect(
                self.server_url,
                ssl=ssl_context,
                subprotocols=["binary"],
                ping_interval=None,
                max_size=10 * 1024 * 1024,
                open_timeout=30
            )
            print(f"✅ 已连接到ASR服务器: {self.server_url}")
            return True

        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    async def send_configuration(self, config: Dict) -> bool:
        """发送配置到服务器"""
        try:
            config_json = json.dumps(config)
            await self.websocket.send(config_json)

            if self.debug_mode:
                print(f"📤 发送的配置JSON:")
                print(f"   {config_json}")

            return True
        except Exception as e:
            print(f"❌ 发送配置失败: {e}")
            return False

    async def start_recognition_simple(self) -> bool:
        """使用最简单的配置开始语音识别"""
        if not self.websocket:
            print("❌ 未连接到服务器")
            return False

        # 尝试多种配置方式
        configs_to_try = [
            {
                "name": "配置1: 最简配置",
                "config": {
                    "is_speaking": True,
                    "wav_name": "test_simple",
                    "mode": "online"
                }
            },
            {
                "name": "配置2: 带chunk_size",
                "config": {
                    "is_speaking": True,
                    "wav_name": "test_with_chunks",
                    "mode": "online",
                    "chunk_size": [5, 10, 5]
                }
            },
            {
                "name": "配置3: offline模式",
                "config": {
                    "is_speaking": True,
                    "wav_name": "test_offline",
                    "mode": "offline"
                }
            },
            {
                "name": "配置4: 完整配置",
                "config": {
                    "is_speaking": True,
                    "wav_name": "test_full",
                    "mode": "online",
                    "chunk_interval": 10,
                    "chunk_size": [5, 10, 5],
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 0
                }
            }
        ]

        for i, config_info in enumerate(configs_to_try):
            print(f"\n尝试 {config_info['name']}...")

            if await self.send_configuration(config_info["config"]):
                self.is_recording = True

                # 等待服务器响应
                response = await self.wait_for_response(timeout=3)
                if response:
                    print(f"✅ 服务器接受了配置: {config_info['name']}")
                    return True
                else:
                    print(f"⚠️  配置 {config_info['name']} 无响应，尝试下一个配置")
                    continue

        print("❌ 所有配置尝试都失败")
        return False

    async def wait_for_response(self, timeout: float = 3.0) -> Optional[str]:
        """等待服务器响应"""
        try:
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=timeout
            )
            if isinstance(response, str):
                print(f"📥 收到服务器响应: {response}")
                return response
        except asyncio.TimeoutError:
            print(f"⏱️  等待响应超时 ({timeout}秒)")
        except Exception as e:
            print(f"⚠️  等待响应时出错: {e}")

        return None

    def resample_audio(self, audio_data: bytes, original_rate: int, target_rate: int = 16000) -> tuple:
        """重采样音频到目标采样率"""
        if original_rate == target_rate:
            return audio_data, target_rate

        if not HAS_SCIPY:
            print(f"❌ 未安装scipy，无法重采样 {original_rate}Hz -> {target_rate}Hz")
            return audio_data, original_rate

        try:
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 计算重采样因子
            num_samples = int(len(audio_array) * target_rate / original_rate)

            # 重采样
            resampled_audio = signal.resample(audio_array, num_samples)

            # 转换回16-bit PCM
            resampled_audio_int16 = (resampled_audio * 32767).astype(np.int16)

            print(f"✅ 音频重采样: {original_rate}Hz -> {target_rate}Hz")
            print(f"   原始样本数: {len(audio_array)}, 重采样后: {len(resampled_audio_int16)}")

            return resampled_audio_int16.tobytes(), target_rate

        except Exception as e:
            print(f"❌ 重采样失败: {e}")
            return audio_data, original_rate

    def create_realistic_speech(self, text: str = "你好世界") -> tuple:
        """
        创建更真实的语音模拟
        模拟中文语音的频率特征
        """
        sample_rate = 16000
        duration_per_char = 0.3  # 每个字符0.3秒
        silence_between_chars = 0.1  # 字符间静音0.1秒

        # 模拟不同中文字符的频率
        char_frequencies = {
            '你': [200, 600, 1200],  # 基频 + 两个泛音
            '好': [250, 750, 1500],
            '世': [300, 900, 1800],
            '界': [280, 840, 1680],
            '测': [260, 780, 1560],
            '试': [270, 810, 1620]
        }

        # 生成音频
        audio_segments = []

        for i, char in enumerate(text):
            if char in char_frequencies:
                freqs = char_frequencies[char]
                num_samples = int(sample_rate * duration_per_char)
                t = np.linspace(0, duration_per_char, num_samples, False)

                # 生成多个频率的混合
                char_audio = np.zeros(num_samples)
                for j, freq in enumerate(freqs):
                    amplitude = 1.0 / (j + 1)  # 泛音幅度递减
                    char_audio += amplitude * np.sin(2 * np.pi * freq * t)

                # 添加包络
                envelope = np.ones_like(t)
                attack = int(0.05 * sample_rate)
                release = int(0.08 * sample_rate)

                if attack > 0:
                    envelope[:attack] = np.linspace(0, 1, attack)
                if release > 0:
                    envelope[-release:] = np.linspace(1, 0, release)

                char_audio = char_audio * envelope * 0.6
                audio_segments.append(char_audio)

            # 添加字符间静音（除了最后一个字符）
            if i < len(text) - 1:
                silence_samples = int(sample_rate * silence_between_chars)
                audio_segments.append(np.zeros(silence_samples))

        # 合并所有段
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
        else:
            # 默认生成一个简单的音频
            duration = 1.0
            num_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, num_samples, False)
            full_audio = 0.7 * np.sin(2 * np.pi * 200 * t)

        # 添加整体包络
        total_samples = len(full_audio)
        overall_envelope = np.ones(total_samples)
        overall_attack = int(0.1 * sample_rate)
        overall_release = int(0.2 * sample_rate)

        if overall_attack > 0:
            overall_envelope[:overall_attack] = np.linspace(0, 1, overall_attack)
        if overall_release > 0:
            overall_envelope[-overall_release:] = np.linspace(1, 0, overall_release)

        full_audio = full_audio * overall_envelope

        # 转换为16-bit PCM
        audio_int16 = (full_audio * 32767).astype(np.int16)

        duration = len(audio_int16) / sample_rate
        print(f"🎵 生成模拟语音: '{text}'")
        print(f"   时长: {duration:.2f}秒, 采样率: {sample_rate}Hz")

        return audio_int16.tobytes(), sample_rate

    async def send_audio_stream(self, audio_data: bytes, sample_rate: int = 16000, chunk_duration_ms: int = 10):
        """以流式方式发送音频数据"""
        if not self.is_recording or not self.websocket:
            print("❌ 未开始识别")
            return False

        try:
            # 确保采样率是16000Hz
            if sample_rate != 16000:
                print(f"⚠️  采样率不是16000Hz: {sample_rate}Hz")
                if HAS_SCIPY:
                    print("🔄 正在重采样到16000Hz...")
                    audio_data, sample_rate = self.resample_audio(audio_data, sample_rate, 16000)
                else:
                    print("❌ 无法重采样，继续使用当前采样率")

            # 计算块大小
            chunk_size = int(sample_rate * chunk_duration_ms / 1000) * 2  # 16-bit = 2字节
            total_chunks = len(audio_data) // chunk_size

            print(f"📤 开始流式发送音频:")
            print(f"  总大小: {len(audio_data)} 字节")
            print(f"  采样率: {sample_rate} Hz")
            print(f"  块大小: {chunk_size} 字节 ({chunk_duration_ms}ms)")
            print(f"  总块数: {total_chunks}")
            print(f"  总时长: {len(audio_data) / (sample_rate * 2):.2f} 秒")

            sent_chunks = 0
            start_time = time.time()

            # 创建异步任务来接收消息
            receive_task = asyncio.create_task(self.continuous_receive(timeout=30))

            for i in range(total_chunks):
                if not self.is_recording:
                    print("⏹️ 识别已停止，中断发送")
                    break

                start = i * chunk_size
                end = start + chunk_size

                if end > len(audio_data):
                    break

                chunk = audio_data[start:end]

                try:
                    # 发送音频块
                    await self.websocket.send(chunk)
                    sent_chunks += 1

                    # 控制发送速率
                    await asyncio.sleep(chunk_duration_ms / 1000)  # 精确控制时间

                    # 显示进度
                    if sent_chunks % 20 == 0:  # 每200ms显示一次
                        elapsed = time.time() - start_time
                        progress = sent_chunks / total_chunks * 100
                        print(f"  进度: {progress:.1f}% ({sent_chunks}/{total_chunks} 块)")

                except websockets.exceptions.ConnectionClosed:
                    print("🔌 连接已关闭，停止发送")
                    return False
                except Exception as e:
                    print(f"⚠️  发送第{i}块时出错: {e}")
                    await asyncio.sleep(0.05)

            print(f"✅ 音频发送完成: {sent_chunks}/{total_chunks} 块")

            # 等待接收任务完成
            print("⏳ 等待服务器处理音频...")
            await asyncio.sleep(2)

            # 停止接收任务
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

            return True

        except Exception as e:
            print(f"❌ 发送音频时发生错误: {e}")
            return False

    async def continuous_receive(self, timeout: int = 30):
        """持续接收服务器消息"""
        print("👂 开始持续监听服务器消息...")

        start_time = time.time()
        last_message_time = start_time

        try:
            while time.time() - start_time < timeout:
                try:
                    # 设置接收超时
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0
                    )

                    last_message_time = time.time()

                    if isinstance(message, str):
                        try:
                            result = json.loads(message)
                            self.results_received.append(result)
                            self.handle_result(result, verbose=True)
                        except json.JSONDecodeError:
                            print(f"📥 收到非JSON消息: {message[:100]}...")
                    else:
                        print(f"📦 收到二进制消息，长度: {len(message)} 字节")

                except asyncio.TimeoutError:
                    # 检查是否长时间没有消息
                    if time.time() - last_message_time > 5:
                        print("⏱️  5秒内未收到消息，继续等待...")
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"🔌 连接已关闭: {e}")
                    break
                except Exception as e:
                    print(f"⚠️  接收消息时出错: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            print(f"❌ 持续接收时发生错误: {e}")

        print(f"📊 监听结束，共收到 {len(self.results_received)} 条消息")

    def handle_result(self, result: Dict, verbose: bool = True):
        """处理识别结果"""
        mode = result.get("mode", "")
        text = result.get("text", "")
        is_final = result.get("is_final", False)
        wav_name = result.get("wav_name", "")

        status = "🔴" if is_final else "🟡"
        timestamp = time.strftime("%H:%M:%S", time.localtime())

        if verbose:
            print(f"[{timestamp}] {status} [{mode}] [{wav_name}]")
            print(f"   识别结果: '{text}'")
            print(f"   完整结果: {result}")
        else:
            print(f"[{timestamp}] {status} [{mode}] [{wav_name}] 识别结果: '{text}'")

    async def send_stop_signal(self):
        """发送停止信号"""
        try:
            stop_msg = {"is_speaking": False}
            await self.websocket.send(json.dumps(stop_msg))
            print(f"📤 已发送停止信号: {stop_msg}")

            # 等待服务器发送最终结果
            print("⏳ 等待服务器发送最终结果...")
            await asyncio.sleep(3)

        except Exception as e:
            print(f"❌ 发送停止信号失败: {e}")

    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
        print("🔌 连接已断开")

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"共收到 {len(self.results_received)} 条消息")

        if self.results_received:
            print("\n收到的所有消息:")
            for i, result in enumerate(self.results_received, 1):
                mode = result.get("mode", "unknown")
                text = result.get("text", "")
                is_final = result.get("is_final", False)
                wav_name = result.get("wav_name", "")

                final_str = "最终" if is_final else "中间"
                print(f"{i:2d}. [{final_str}] [{mode}] [{wav_name}] '{text}'")

                # 如果有详细数据，也显示
                if self.debug_mode and len(result) > 4:
                    for key, value in result.items():
                        if key not in ["mode", "text", "is_final", "wav_name"]:
                            print(f"     {key}: {value}")
        else:
            print("⚠️  未收到任何识别结果")
            print("\n可能的原因:")
            print("1. 服务器ASR模型未正确加载")
            print("2. 音频格式或内容不符合要求")
            print("3. VAD未检测到语音活动")
            print("4. 服务器配置有问题")

        print("=" * 60)


async def test_diagnostic():
    """诊断测试 - 找出问题所在"""
    print("🔍 FunASR 诊断测试")
    print("=" * 60)

    # 创建客户端
    client = ASRClientPython(server_url="wss://192.168.21.130:10095/")
    client.debug_mode = True

    try:
        # 1. 测试连接
        print("1. 测试连接...")
        if not await client.connect(verify_cert=False):
            print("❌ 连接失败，退出测试")
            return

        # 2. 测试配置接收
        print("\n2. 测试配置接收...")
        print("   发送简单配置，等待服务器响应...")

        simple_config = {
            "is_speaking": True,
            "wav_name": "diagnostic_test",
            "mode": "online"
        }

        await client.send_configuration(simple_config)

        # 等待响应
        response = await client.wait_for_response(timeout=5)
        if response:
            print("✅ 服务器响应了配置")
            try:
                resp_json = json.loads(response)
                print(f"   响应内容: {resp_json}")
            except:
                print(f"   响应内容(原始): {response}")
        else:
            print("⚠️  服务器未响应配置，但继续测试")

        # 3. 发送测试音频
        print("\n3. 发送测试音频...")
        client.is_recording = True

        # 生成更真实的测试音频
        audio_data, sample_rate = client.create_realistic_speech("你好世界测试")

        # 发送音频
        success = await client.send_audio_stream(audio_data, sample_rate, chunk_duration_ms=10)

        if not success:
            print("❌ 音频发送失败")

        # 4. 发送停止信号
        print("\n4. 发送停止信号...")
        await client.send_stop_signal()

        # 5. 额外等待
        print("\n5. 额外等待服务器响应...")
        await asyncio.sleep(5)

        # 6. 断开连接
        await client.disconnect()

        # 7. 打印总结
        client.print_summary()

        print("\n✅ 诊断测试完成")

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断测试")
        await client.disconnect()
    except Exception as e:
        print(f"\n❌ 诊断测试失败: {e}")
        import traceback
        traceback.print_exc()
        await client.disconnect()


async def test_with_real_audio():
    """使用真实音频文件测试"""
    print("🎵 使用真实音频文件测试")
    print("=" * 60)

    # 创建客户端
    client = ASRClientPython(server_url="wss://192.168.21.130:10095/")

    # 查找音频文件
    audio_files = [
        "test.wav",
        "test_simple.wav",
        "test_speech.wav",
        "audio.wav",
        "sample.wav"
    ]

    found_file = None
    for file in audio_files:
        if os.path.exists(file):
            found_file = file
            break

    if not found_file:
        print("❌ 未找到任何测试音频文件")
        print("   请将音频文件放置在当前目录下，文件名可以是:")
        print("   - test.wav")
        print("   - test_simple.wav")
        print("   - test_speech.wav")
        print("   - audio.wav")
        print("   - sample.wav")
        return

    print(f"🎵 找到音频文件: {found_file}")

    try:
        # 1. 连接服务器
        print("\n1. 连接服务器...")
        if not await client.connect(verify_cert=False):
            print("❌ 连接失败，退出测试")
            return

        # 2. 加载音频文件
        print("\n2. 加载音频文件...")
        try:
            with wave.open(found_file, 'rb') as wav_file:
                params = wav_file.getparams()
                original_rate = params.framerate
                print(f"   采样率: {original_rate} Hz")
                print(f"   声道数: {params.nchannels}")
                print(f"   位深度: {params.sampwidth * 8} bit")
                print(f"   帧数: {params.nframes}")
                print(f"   时长: {params.nframes / original_rate:.2f} 秒")

                # 读取音频数据
                audio_data = wav_file.readframes(params.nframes)

                # 如果是立体声，转换为单声道
                if params.nchannels == 2:
                    print("   🔄 将立体声转换为单声道...")
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    audio_data = audio_array.tobytes()

        except Exception as e:
            print(f"❌ 加载音频文件失败: {e}")
            return

        # 3. 发送配置 - 使用更完整的配置
        print("\n3. 发送配置...")
        config = {
            "is_speaking": True,
            "wav_name": os.path.basename(found_file),
            "mode": "online",
            "chunk_size": [5, 10, 5],  # 添加chunk_size
            "encoder_chunk_look_back": 4,  # 添加look_back参数
            "decoder_chunk_look_back": 0
        }

        await client.send_configuration(config)
        client.is_recording = True

        # 4. 发送音频（会自动重采样到16000Hz）
        print("\n4. 发送音频...")
        success = await client.send_audio_stream(audio_data, original_rate, chunk_duration_ms=10)

        # 5. 停止识别
        print("\n5. 停止识别...")
        await client.send_stop_signal()

        # 6. 额外等待服务器响应
        print("\n6. 额外等待服务器响应...")
        await asyncio.sleep(5)

        # 7. 断开连接
        await client.disconnect()

        # 8. 打印总结
        client.print_summary()

        print("\n✅ 真实音频测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        await client.disconnect()


def create_test_audio():
    """创建高质量的测试音频"""
    try:
        import wave
        import numpy as np

        sample_rate = 16000
        duration = 3.0  # 3秒

        print("🎵 创建高质量测试音频...")

        # 模拟一段中文语音："大家好，这是一个测试"
        # 每个字的持续时间和频率
        syllables = [
            {"char": "大", "duration": 0.3, "freqs": [200, 600, 1200]},
            {"char": "家", "duration": 0.3, "freqs": [220, 660, 1320]},
            {"char": "好", "duration": 0.4, "freqs": [180, 540, 1080]},
            {"char": "这", "duration": 0.2, "freqs": [250, 750, 1500]},
            {"char": "是", "duration": 0.2, "freqs": [240, 720, 1440]},
            {"char": "一", "duration": 0.3, "freqs": [210, 630, 1260]},
            {"char": "个", "duration": 0.2, "freqs": [230, 690, 1380]},
            {"char": "测", "duration": 0.3, "freqs": [260, 780, 1560]},
            {"char": "试", "duration": 0.4, "freqs": [190, 570, 1140]}
        ]

        # 生成音频段
        audio_segments = []

        for syllable in syllables:
            # 生成音节
            num_samples = int(sample_rate * syllable["duration"])
            t = np.linspace(0, syllable["duration"], num_samples, False)

            syllable_audio = np.zeros(num_samples)
            for i, freq in enumerate(syllable["freqs"]):
                amplitude = 0.7 / (i + 1)
                syllable_audio += amplitude * np.sin(2 * np.pi * freq * t)

            # 音节包络
            envelope = np.ones(num_samples)
            attack = int(0.05 * sample_rate)
            release = int(0.08 * sample_rate)

            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release > 0:
                envelope[-release:] = np.linspace(1, 0, release)

            syllable_audio = syllable_audio * envelope
            audio_segments.append(syllable_audio)

            # 音节间添加短暂静音（除了最后一个）
            if syllable != syllables[-1]:
                silence_duration = 0.05  # 50ms静音
                silence_samples = int(sample_rate * silence_duration)
                audio_segments.append(np.zeros(silence_samples))

        # 合并所有段
        full_audio = np.concatenate(audio_segments)

        # 如果总时长不够，用静音补齐
        target_samples = int(sample_rate * duration)
        if len(full_audio) < target_samples:
            silence_needed = target_samples - len(full_audio)
            full_audio = np.concatenate([full_audio, np.zeros(silence_needed)])
        elif len(full_audio) > target_samples:
            full_audio = full_audio[:target_samples]

        # 整体包络
        overall_envelope = np.ones(len(full_audio))
        overall_attack = int(0.1 * sample_rate)
        overall_release = int(0.2 * sample_rate)

        if overall_attack > 0:
            overall_envelope[:overall_attack] = np.linspace(0, 1, overall_attack)
        if overall_release > 0:
            overall_envelope[-overall_release:] = np.linspace(1, 0, overall_release)

        full_audio = full_audio * overall_envelope

        # 转换为16-bit PCM
        audio_int16 = (full_audio * 32767).astype(np.int16)

        # 写入WAV文件
        filename = "test_high_quality.wav"
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)  # 单声道
            wav.setsampwidth(2)  # 16-bit = 2字节
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        print(f"✅ 已创建高质量测试音频: {filename}")
        print(f"   时长: {duration}秒, 采样率: {sample_rate}Hz")
        print(f"   模拟语音: '大家好，这是一个测试'")

        return filename

    except Exception as e:
        print(f"❌ 创建测试音频失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("FunASR 语音识别测试工具")
    print("=" * 60)

    if not HAS_SCIPY:
        print("⚠️  建议安装scipy进行音频重采样:")
        print("    pip install scipy")
        print()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--diagnostic":
            asyncio.run(test_diagnostic())
        elif sys.argv[1] == "--real-audio":
            asyncio.run(test_with_real_audio())
        elif sys.argv[1] == "--create-audio":
            create_test_audio()
        else:
            print("用法:")
            print("  python test_doc.py --diagnostic    运行诊断测试")
            print("  python test_doc.py --real-audio    使用真实音频文件测试")
            print("  python test_doc.py --create-audio  创建高质量测试音频")
    else:
        # 默认运行诊断测试
        asyncio.run(test_diagnostic())


# # 运行真实音频测试（会自动重采样）
# python test_doc.py --real-audio
#
# # 运行诊断测试
# python test_doc.py --diagnostic
#
# # 创建高质量的测试音频
# python test_doc.py --create-audio