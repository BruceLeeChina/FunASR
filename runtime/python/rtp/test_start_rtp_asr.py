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


class RTPStreamProcessor:
    def __init__(self, rtp_url, websocket):
        self.rtp_url = rtp_url
        self.websocket = websocket
        self.is_running = False
        self.ffmpeg_process = None
        self.audio_buffer = bytearray()
        self.chunk_size = 320  # 10ms of 16kHz 16-bit audio
        self.total_bytes_received = 0

    def start(self):
        """启动RTP流处理"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._process_rtp_stream)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"RTP stream processor started for {self.rtp_url}")

    def stop(self):
        """停止RTP流处理"""
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
        logger.info(f"RTP stream processor stopped for {self.rtp_url}")

    def _process_rtp_stream(self):
        """处理RTP流并模拟WebSocket音频消息"""
        try:
            logger.info(f"Starting RTP stream processing for {self.rtp_url}")

            # 使用FFmpeg接收RTP流并转换为16kHz PCM
            ffmpeg_cmd = [
                'ffmpeg',
                '-loglevel', 'error',
                '-protocol_whitelist', 'file,udp,rtp',
                '-i', self.rtp_url,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-f', 's16le',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-'
            ]

            logger.info(f"Starting FFmpeg: {' '.join(ffmpeg_cmd)}")

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # 创建事件循环来处理WebSocket消息
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def run_async_in_thread():
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._audio_processing_loop())

            processing_thread = threading.Thread(target=run_async_in_thread)
            processing_thread.daemon = True
            processing_thread.start()

            consecutive_errors = 0
            max_errors = 10

            while self.is_running:
                try:
                    # 读取音频数据
                    data = self.ffmpeg_process.stdout.read(1024)
                    if data:
                        self.audio_buffer.extend(data)
                        self.total_bytes_received += len(data)
                        consecutive_errors = 0
                    else:
                        if self.ffmpeg_process.poll() is not None:
                            logger.error("FFmpeg process terminated")
                            stderr = self.ffmpeg_process.stderr.read()
                            if stderr:
                                logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
                            break
                        time.sleep(0.01)

                except Exception as e:
                    logger.error(f"Error reading from FFmpeg: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        break
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in RTP stream processing: {e}")
        finally:
            self.is_running = False

    async def _audio_processing_loop(self):
        """处理音频数据并模拟WebSocket消息"""
        chunk_count = 0
        last_process_time = time.time()

        while self.is_running:
            try:
                current_time = time.time()

                # 每10ms处理一次（模拟实时音频流）
                if current_time - last_process_time >= 0.01 and len(self.audio_buffer) >= self.chunk_size:
                    # 提取一个音频块
                    audio_chunk = bytes(self.audio_buffer[:self.chunk_size])
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]

                    # 模拟WebSocket音频消息处理
                    await self._process_audio_chunk(audio_chunk)

                    chunk_count += 1
                    if chunk_count % 100 == 0:  # 每1秒记录一次
                        logger.info(f"Processed {chunk_count} audio chunks, total bytes: {self.total_bytes_received}")

                    last_process_time = current_time
                else:
                    await asyncio.sleep(0.001)  # 短暂休眠

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_audio_chunk(self, audio_chunk):
        """处理单个音频块 - 模拟原始WebSocket处理逻辑"""
        try:
            # 这里模拟原始funasr_wss_server.py的处理逻辑
            websocket = self.websocket

            # 添加到在线ASR缓冲区
            if not hasattr(websocket, 'frames_asr_online'):
                websocket.frames_asr_online = []
            if not hasattr(websocket, 'frames_asr'):
                websocket.frames_asr = []

            websocket.frames_asr_online.append(audio_chunk)

            # 模拟语音活动（假设一直在说话）
            if not hasattr(websocket, 'speech_start'):
                websocket.speech_start = True
            if not hasattr(websocket, 'is_speaking'):
                websocket.is_speaking = True

            if websocket.speech_start:
                websocket.frames_asr.append(audio_chunk)

            # 定期触发在线ASR（每10个块触发一次）
            if len(websocket.frames_asr_online) >= 10:  # 相当于100ms的音频
                if websocket.mode == "2pass" or websocket.mode == "online":
                    audio_in = b"".join(websocket.frames_asr_online)
                    if len(audio_in) > 0:
                        try:
                            await self._async_asr_online(websocket, audio_in)
                        except Exception as e:
                            logger.error(f"Error in online ASR: {e}")
                    websocket.frames_asr_online = []

            # VAD检测（简化版，假设检测到语音）
            if not hasattr(websocket, 'vad_pre_idx'):
                websocket.vad_pre_idx = 0
            websocket.vad_pre_idx += len(audio_chunk) // 32

            # 模拟VAD检测到语音开始
            if not hasattr(websocket, 'speech_start_i_called'):
                websocket.speech_start_i_called = True
                logger.info("VAD detected speech start")

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def _async_asr_online(self, websocket, audio_in):
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
                    "wav_name": getattr(websocket, 'wav_name', 'rtp_stream'),
                    "is_final": getattr(websocket, 'is_speaking', False),
                })
                logger.info(f"Online ASR result: {rec_result['text']}")
                await websocket.send(message)

        except Exception as e:
            logger.error(f"Error in online ASR: {e}")


async def ws_reset(websocket):
    logger.info(f"WebSocket reset, total users: {len(websocket_users)}")

    # 重置状态
    if hasattr(websocket, 'status_dict_asr_online'):
        websocket.status_dict_asr_online["cache"] = {}
        websocket.status_dict_asr_online["is_final"] = True
    if hasattr(websocket, 'status_dict_vad'):
        websocket.status_dict_vad["cache"] = {}
        websocket.status_dict_vad["is_final"] = True
    if hasattr(websocket, 'status_dict_punc'):
        websocket.status_dict_punc["cache"] = {}

    # 停止RTP处理器
    if hasattr(websocket, 'rtp_processor') and websocket.rtp_processor:
        websocket.rtp_processor.stop()

    try:
        await websocket.close()
    except:
        pass


async def websocket_handler(websocket, path=None):
    """WebSocket处理函数 - 基于原始funasr_wss_server.py的逻辑"""
    global websocket_users

    websocket_users.add(websocket)

    # 初始化状态字典（与原始代码保持一致）
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    websocket.speech_start = False
    websocket.speech_end_i = -1
    websocket.wav_name = "rtp_stream"
    websocket.mode = "2pass"
    websocket.is_speaking = False
    websocket.rtp_processor = None

    # 音频缓冲区
    websocket.frames_asr_online = []
    websocket.frames_asr = []

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

                    # 停止之前的RTP处理器
                    if websocket.rtp_processor:
                        websocket.rtp_processor.stop()

                    # 创建新的RTP处理器
                    websocket.rtp_processor = RTPStreamProcessor(rtp_url, websocket)
                    websocket.rtp_processor.start()

                # 处理其他配置（与原始代码保持一致）
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

                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name")

    except websockets.ConnectionClosed:
        logger.info(f"Connection closed from {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Exception in websocket_handler: {e}")
    finally:
        await ws_reset(websocket)
        if websocket in websocket_users:
            websocket_users.remove(websocket)


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