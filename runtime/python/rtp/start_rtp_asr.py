import asyncio
import json
import websockets
import time
import logging
import numpy as np
import argparse
import ssl
import threading
import queue
import subprocess
import os
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="127.0.0.1", required=False, help="host ip, localhost, 127.0.0.1"
)
parser.add_argument("--port", type=int, default=10097, required=False, help="websocket server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    help="model from modelscope",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="model from modelscope",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")

args = parser.parse_args()

websocket_users = set()

print("model loading")
from funasr import AutoModel

# ASR models
try:
    model_asr = AutoModel(
        model=args.asr_model,
        model_revision=args.asr_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )

    model_asr_streaming = AutoModel(
        model=args.asr_model_online,
        model_revision=args.asr_model_online_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )

    # VAD model
    model_vad = AutoModel(
        model=args.vad_model,
        model_revision=args.vad_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )

    # Punctuation model
    if args.punc_model != "":
        model_punc = AutoModel(
            model=args.punc_model,
            model_revision=args.punc_model_revision,
            ngpu=args.ngpu,
            ncpu=args.ncpu,
            device=args.device,
            disable_pbar=True,
            disable_log=True,
        )
    else:
        model_punc = None

    print("model loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)


class RTPStreamReader:
    def __init__(self, rtp_url, websocket):
        self.rtp_url = rtp_url
        self.websocket = websocket
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.thread = None
        self.ffmpeg_process = None
        self.total_audio_bytes = 0
        self.packet_count = 0
        self.audio_buffer = bytearray()
        self.buffer_size = 3200  # 100ms of 16kHz 16-bit audio

    def start(self):
        """启动RTP流读取线程"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._read_rtp_stream)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"RTP stream reader started for {self.rtp_url}")

    def stop(self):
        """停止RTP流读取"""
        self.is_running = False
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"RTP stream reader stopped for {self.rtp_url}")

    def _read_rtp_stream(self):
        """使用FFmpeg读取RTP流"""
        try:
            logger.info(f"Starting RTP stream reader for {self.rtp_url}")

            # 使用FFmpeg命令行工具接收RTP流
            # 将pcm_mulaw 8kHz转换为pcm_s16le 16kHz
            ffmpeg_cmd = [
                'ffmpeg',
                '-loglevel', 'error',  # 减少日志输出
                '-protocol_whitelist', 'file,udp,rtp',
                '-i', self.rtp_url,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-f', 's16le',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-avioflags', 'direct',
                '-'
            ]

            logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            consecutive_errors = 0
            max_errors = 10

            while self.is_running:
                try:
                    # 读取音频数据，使用较小的读取大小
                    audio_data = self.ffmpeg_process.stdout.read(1024)  # 读取1KB

                    if audio_data:
                        self.audio_buffer.extend(audio_data)
                        self.total_audio_bytes += len(audio_data)

                        # 当缓冲区有足够数据时，放入队列
                        while len(self.audio_buffer) >= self.buffer_size:
                            chunk = bytes(self.audio_buffer[:self.buffer_size])
                            self._add_to_queue(chunk)
                            self.audio_buffer = self.audio_buffer[self.buffer_size:]
                            self.packet_count += 1

                            if self.packet_count % 50 == 0:
                                logger.info(
                                    f"Processed {self.packet_count} packets, total audio bytes: {self.total_audio_bytes}")
                                logger.info(f"Audio queue size: {self.audio_queue.qsize()}")

                        consecutive_errors = 0
                    else:
                        # 检查进程是否还在运行
                        if self.ffmpeg_process.poll() is not None:
                            logger.error("FFmpeg process has terminated")
                            # 读取错误输出
                            stderr_output = self.ffmpeg_process.stderr.read()
                            if stderr_output:
                                logger.error(f"FFmpeg stderr: {stderr_output.decode('utf-8', errors='ignore')}")
                            break

                        # 短暂休眠避免CPU占用过高
                        time.sleep(0.001)

                except Exception as e:
                    logger.error(f"Error reading from FFmpeg: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logger.error("Too many consecutive errors, stopping RTP reader")
                        break
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in RTP stream reader: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass

    def _add_to_queue(self, audio_bytes):
        """添加音频数据到队列"""
        try:
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            # 队列满时丢弃最旧的数据
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_bytes)
            except:
                pass

    def get_audio_chunk(self, chunk_size=3200):
        """获取音频数据块"""
        # 如果队列为空，返回None而不是静音，让上层处理
        if self.audio_queue.empty():
            return None

        chunks = []
        total_size = 0

        while not self.audio_queue.empty() and total_size < chunk_size:
            try:
                chunk = self.audio_queue.get_nowait()
                chunks.append(chunk)
                total_size += len(chunk)
            except queue.Empty:
                break

        if chunks and total_size > 0:
            combined_chunk = b''.join(chunks)
            # 如果数据不足，用静音填充到期望大小
            if len(combined_chunk) < chunk_size:
                silence_needed = chunk_size - len(combined_chunk)
                silence = b'\x00' * silence_needed
                combined_chunk += silence
                logger.debug(f"Padded audio chunk with {silence_needed} bytes of silence")

            return combined_chunk
        else:
            return None


async def ws_reset(websocket):
    logger.info(f"ws reset now, total users: {len(websocket_users)}")

    # 重置状态
    if hasattr(websocket, 'status_dict_asr_online'):
        websocket.status_dict_asr_online["cache"] = {}
        websocket.status_dict_asr_online["is_final"] = True
    if hasattr(websocket, 'status_dict_vad'):
        websocket.status_dict_vad["cache"] = {}
        websocket.status_dict_vad["is_final"] = True
    if hasattr(websocket, 'status_dict_punc'):
        websocket.status_dict_punc["cache"] = {}

    # 停止RTP流
    if hasattr(websocket, 'rtp_reader') and websocket.rtp_reader:
        websocket.rtp_reader.stop()
        del websocket.rtp_reader

    try:
        await websocket.close()
    except:
        pass


async def clear_websocket():
    for websocket in websocket_users.copy():
        await ws_reset(websocket)
    websocket_users.clear()


async def websocket_handler(websocket, path=None):
    """WebSocket handler"""
    global websocket_users

    websocket_users.add(websocket)

    # 初始化websocket属性
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    websocket.chunk_interval = 5  # 减少chunk_interval以获得更快的响应
    websocket.vad_pre_idx = 0
    websocket.speech_start = False
    websocket.speech_end_i = -1
    websocket.wav_name = "rtp_stream"
    websocket.mode = "2pass"
    websocket.rtp_reader = None
    websocket.frames_asr_online = deque(maxlen=50)  # 减少缓冲区大小
    websocket.frames_asr = deque(maxlen=100)
    websocket.is_speaking = False
    websocket.last_process_time = time.time()
    websocket.last_audio_time = 0
    websocket.audio_chunk_count = 0
    websocket.consecutive_no_audio = 0
    websocket.audio_buffer = bytearray()

    logger.info(f"New user connected from {websocket.remote_address}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    messagejson = json.loads(message)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue

                # 处理RTP地址设置
                if "rtp_url" in messagejson:
                    rtp_url = messagejson["rtp_url"]
                    logger.info(f"Setting RTP URL: {rtp_url}")

                    # 停止之前的RTP流
                    if websocket.rtp_reader:
                        websocket.rtp_reader.stop()

                    # 创建新的RTP读取器
                    websocket.rtp_reader = RTPStreamReader(rtp_url, websocket)
                    websocket.rtp_reader.start()

                # 处理其他配置
                if "is_speaking" in messagejson:
                    websocket.is_speaking = messagejson["is_speaking"]
                    websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                    logger.info(f"Set is_speaking to {websocket.is_speaking}")

                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = messagejson["chunk_interval"]

                if "chunk_size" in messagejson:
                    chunk_size = messagejson["chunk_size"]
                    if isinstance(chunk_size, str):
                        chunk_size = chunk_size.split(",")
                    websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]

                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]
                    logger.info(f"Set mode to {websocket.mode}")

                if "hotwords" in messagejson:
                    websocket.status_dict_asr["hotword"] = messagejson["hotwords"]

            # 定期处理音频数据
            current_time = time.time()
            if current_time - websocket.last_process_time >= 0.1:  # 每100ms处理一次
                await process_audio_data(websocket)
                websocket.last_process_time = current_time

    except websockets.ConnectionClosed:
        logger.info(f"Connection closed from {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Exception in websocket_handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ws_reset(websocket)
        if websocket in websocket_users:
            websocket_users.remove(websocket)


async def process_audio_data(websocket):
    """处理音频数据"""
    try:
        if not websocket.rtp_reader or not websocket.rtp_reader.is_running:
            return

        # 获取音频数据
        audio_chunk = websocket.rtp_reader.get_audio_chunk(3200)

        if audio_chunk is not None and len(audio_chunk) > 0:
            websocket.consecutive_no_audio = 0
            websocket.audio_chunk_count += 1

            logger.info(f"Processing audio chunk #{websocket.audio_chunk_count}, size: {len(audio_chunk)} bytes")
            await process_audio_chunk(websocket, audio_chunk)
        else:
            websocket.consecutive_no_audio += 1
            if websocket.consecutive_no_audio == 1:
                logger.debug("No audio data available")
            elif websocket.consecutive_no_audio % 50 == 0:  # 每5秒记录一次
                logger.warning(f"No audio data received for {websocket.consecutive_no_audio * 0.1} seconds")

    except Exception as e:
        logger.error(f"Error in process_audio_data: {e}")
        import traceback
        traceback.print_exc()


async def process_audio_chunk(websocket, audio_chunk):
    """处理音频数据块"""
    try:
        # 计算音频时长
        duration_ms = len(audio_chunk) // 32  # 16kHz 16bit 单声道：每32字节 = 1ms
        websocket.vad_pre_idx += duration_ms

        # 在线ASR处理
        websocket.frames_asr_online.append(audio_chunk)
        websocket.status_dict_asr_online["is_final"] = websocket.speech_end_i != -1

        # 定期触发在线ASR
        should_process_online = (
                len(websocket.frames_asr_online) >= websocket.chunk_interval or
                websocket.status_dict_asr_online["is_final"]
        )

        if should_process_online:
            logger.info(f"Triggering online ASR processing, frames: {len(websocket.frames_asr_online)}")
            if websocket.mode in ["2pass", "online"]:
                audio_in = b"".join(websocket.frames_asr_online)
                if len(audio_in) > 0:
                    logger.info(f"Online ASR input: {len(audio_in)} bytes")
                    try:
                        await async_asr_online(websocket, audio_in)
                    except Exception as e:
                        logger.error(f"Error in online ASR: {e}")
                websocket.frames_asr_online.clear()

        # VAD检测
        try:
            speech_start_i, speech_end_i = await async_vad(websocket, audio_chunk)
            if speech_start_i != -1 or speech_end_i != -1:
                logger.info(f"VAD result: start={speech_start_i}, end={speech_end_i}")

            if speech_start_i != -1:
                websocket.speech_start = True
                logger.info("Speech start detected")

            if speech_end_i != -1:
                websocket.speech_end_i = speech_end_i
                logger.info("Speech end detected")
        except Exception as e:
            logger.error(f"Error in VAD: {e}")

        # 离线ASR处理（语音结束时）
        if websocket.speech_start:
            websocket.frames_asr.append(audio_chunk)

        should_process_offline = (
                websocket.speech_end_i != -1 or
                not websocket.is_speaking
        )

        if should_process_offline and len(websocket.frames_asr) > 0:
            logger.info("Triggering offline ASR processing")
            if websocket.mode in ["2pass", "offline"]:
                audio_in = b"".join(websocket.frames_asr)
                if len(audio_in) > 0:
                    logger.info(f"Offline ASR input: {len(audio_in)} bytes")
                    try:
                        await async_asr(websocket, audio_in)
                    except Exception as e:
                        logger.error(f"Error in offline ASR: {e}")
                websocket.frames_asr.clear()
                websocket.speech_start = False
                websocket.speech_end_i = -1

    except Exception as e:
        logger.error(f"Error in process_audio_chunk: {e}")
        import traceback
        traceback.print_exc()


async def async_vad(websocket, audio_in):
    """VAD语音活动检测"""
    try:
        if len(audio_in) == 0:
            return -1, -1

        # VAD处理
        result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)

        if not result or len(result) == 0:
            return -1, -1

        vad_result = result[0]
        if "value" not in vad_result:
            return -1, -1

        segments = vad_result["value"]
        if not segments or len(segments) == 0:
            return -1, -1

        # 提取语音段信息
        speech_start = -1
        speech_end = -1

        if isinstance(segments[0], (list, tuple)) and len(segments[0]) >= 2:
            speech_start = segments[0][0] if segments[0][0] != -1 else -1
            speech_end = segments[0][1] if segments[0][1] != -1 else -1

        return speech_start, speech_end

    except Exception as e:
        logger.error(f"Error in VAD processing: {e}")
        return -1, -1


async def async_asr(websocket, audio_in):
    """离线ASR识别"""
    try:
        if len(audio_in) == 0:
            return

        rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)

        if not rec_result or len(rec_result) == 0:
            return

        rec_result = rec_result[0]

        # 标点处理
        if model_punc is not None and len(rec_result.get("text", "")) > 0:
            rec_result = model_punc.generate(
                input=rec_result["text"], **websocket.status_dict_punc
            )[0]

        if len(rec_result.get("text", "")) > 0:
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps({
                "mode": mode,
                "text": rec_result["text"],
                "wav_name": websocket.wav_name,
                "is_final": True,
            })
            logger.info(f"Sending offline ASR result: {rec_result['text']}")
            await websocket.send(message)

    except Exception as e:
        logger.error(f"Error in async_asr: {e}")
        import traceback
        traceback.print_exc()


async def async_asr_online(websocket, audio_in):
    """在线ASR识别"""
    try:
        if len(audio_in) == 0:
            return

        # 在2pass模式下，如果is_final为True，跳过在线识别
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return

        rec_result = model_asr_streaming.generate(
            input=audio_in, **websocket.status_dict_asr_online
        )

        if not rec_result or len(rec_result) == 0:
            return

        rec_result = rec_result[0]

        if len(rec_result.get("text", "")) > 0:
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps({
                "mode": mode,
                "text": rec_result["text"],
                "wav_name": websocket.wav_name,
                "is_final": websocket.is_speaking,
            })
            logger.info(f"Sending online ASR result: {rec_result['text']}")
            await websocket.send(message)

    except Exception as e:
        logger.error(f"Error in async_asr_online: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    if args.certfile and args.keyfile:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.certfile, keyfile=args.keyfile)

        async with websockets.serve(
                websocket_handler, args.host, args.port, ping_interval=None, ssl=ssl_context
        ):
            logger.info(f"WebSocket server started (secure) at wss://{args.host}:{args.port}")
            await asyncio.Future()
    else:
        async with websockets.serve(
                websocket_handler, args.host, args.port, ping_interval=None
        ):
            logger.info(f"WebSocket server started (insecure) at ws://{args.host}:{args.port}")
            await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())