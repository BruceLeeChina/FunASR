import sys
import numpy as np
from funasr import AutoModel
import subprocess
import threading
import time
import queue
from flask import Flask, Response, render_template
import json
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- 配置常量 ---
RTSP_URL = "rtsp://127.0.0.1:8554/test"

# 音频处理参数
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
CHUNK_DURATION = 0.2
CHUNK_SIZE = int(TARGET_SAMPLE_RATE * CHUNK_DURATION * BYTES_PER_SAMPLE)

# 识别参数
BUFFER_DURATION = 5
KEEP_DURATION = 0.5
BUFFER_SIZE = int(TARGET_SAMPLE_RATE * BUFFER_DURATION * BYTES_PER_SAMPLE)
KEEP_SIZE = int(TARGET_SAMPLE_RATE * KEEP_DURATION * BYTES_PER_SAMPLE)

# 全局变量
text_queue = queue.Queue()
recognition_active = False
process = None
model = None
full_text = ""  # 存储完整的识别文本


def init_model():
    """初始化语音识别模型"""
    global model
    logger.info("正在加载模型...")
    try:
        model = AutoModel(
            model="paraformer-zh",
            disable_update=True,
            device="cuda",
            vad_model="fsmn-vad",
            punc_model="ct-punc-c",  # 中文标点模型
        )
        logger.info("模型加载完成。")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        # 如果标点模型加载失败，尝试加载不带标点的模型
        try:
            model = AutoModel(
                model="paraformer-zh",
                disable_update=True,
                device="cuda"
            )
            logger.info("模型加载完成（无标点功能）。")
            return True
        except Exception as e2:
            logger.error(f"备用模型加载失败: {e2}")
            return False


def log_ffmpeg_error(pipe):
    """处理FFmpeg错误输出"""
    for line in pipe:
        line_str = line.decode('utf-8', errors='ignore').strip()
        if line_str:
            logger.info(f"[FFmpeg]: {line_str}")


def start_ffmpeg_process():
    """启动FFmpeg进程"""
    command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-vn", "-nostdin",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-f", "s16le",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "pipe:1"
    ]

    logger.info(f"启动FFmpeg进程: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        return process
    except FileNotFoundError:
        logger.error("错误: 'ffmpeg' 命令未找到。请确保FFmpeg已安装并添加到系统PATH中。")
        return None


def convert_audio_to_float(audio_bytes):
    """将16位PCM音频转换为浮点数格式"""
    # 将字节数据转换为int16数组
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # 转换为float32并归一化到[-1, 1]范围
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return audio_float


def clean_text(text):
    """清理文本，去除多余空格和特殊字符"""
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    text = text.strip()
    return text


def add_punctuation(text):
    """为文本添加标点符号"""
    if not text:
        return text

    # 简单的标点符号添加规则
    # 1. 在疑问词后添加问号
    question_words = ['吗', '呢', '什么', '为什么', '怎么', '如何', '哪', '谁', '何时', '何地']
    for word in question_words:
        if word in text and not text.endswith('？') and not text.endswith('?'):
            text += '？'
            break

    # 2. 如果文本较长且没有标点，在适当位置添加逗号
    if len(text) > 10 and '，' not in text and '。' not in text:
        # 在大概中间位置添加逗号
        mid_pos = len(text) // 2
        # 找到最近的空格位置
        space_pos = text.find(' ', mid_pos)
        if space_pos != -1:
            text = text[:space_pos] + '，' + text[space_pos + 1:]

    # 3. 确保文本以句号结束（如果还没有标点结尾）
    if text and not text[-1] in ['。', '！', '？', '.', '!', '?']:
        text += '。'

    return text


def merge_text(new_text, full_text):
    """将新文本合并到完整文本中"""
    if not new_text:
        return full_text

    # 清理新文本
    new_text = clean_text(new_text)

    # 为新文本添加标点
    new_text = add_punctuation(new_text)

    # 如果完整文本为空，直接返回新文本
    if not full_text:
        return new_text

    # 检查新文本是否已经包含在完整文本中（避免重复）
    if new_text in full_text:
        return full_text

    # 检查是否有重叠部分（例如：完整文本以"今天天气"结尾，新文本是"天气很好"）
    overlap_length = min(len(full_text), len(new_text)) // 2
    if overlap_length > 0:
        for i in range(overlap_length, 0, -1):
            if full_text.endswith(new_text[:i]):
                # 找到重叠部分，合并文本
                return full_text + new_text[i:]

    # 没有重叠，直接追加
    # 如果完整文本以标点结尾，直接追加新文本
    if full_text and full_text[-1] in ['。', '！', '？', '.', '!', '?']:
        return full_text + " " + new_text
    else:
        # 否则用逗号连接
        return full_text + "，" + new_text


def recognition_thread():
    """语音识别主线程"""
    global recognition_active, process, text_queue, full_text

    accumulated_audio = bytes()
    last_text = ""
    reconnect_attempts = 0
    max_reconnect_attempts = 10

    # 使用全局变量 process
    global_process = None

    def setup_ffmpeg():
        """设置FFmpeg进程和错误日志线程"""
        nonlocal global_process  # 使用 nonlocal 引用外部变量
        global_process = start_ffmpeg_process()
        if global_process:
            err_thread = threading.Thread(target=log_ffmpeg_error, args=(global_process.stderr,))
            err_thread.daemon = True
            err_thread.start()
        return global_process

    # 初始启动FFmpeg
    global_process = setup_ffmpeg()

    while recognition_active:
        if global_process is None:
            logger.info("FFmpeg进程未启动，尝试重新启动...")
            global_process = setup_ffmpeg()
            if global_process is None:
                logger.info("无法启动FFmpeg进程，5秒后重试...")
                time.sleep(5)
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error("达到最大重连次数，停止识别线程。")
                    break
                continue

        # 读取音频数据
        try:
            pcm_chunk = global_process.stdout.read(CHUNK_SIZE)
        except Exception as e:
            logger.error(f"读取音频数据时出错: {e}")
            pcm_chunk = None

        # 处理空数据
        if not pcm_chunk:
            if global_process.poll() is not None:
                logger.info("FFmpeg进程已退出，尝试重新启动...")
                global_process = None
                time.sleep(2)
                continue
            else:
                time.sleep(0.1)
                continue

        # 重置重连计数
        reconnect_attempts = 0

        # 累积音频数据
        accumulated_audio += pcm_chunk

        # 当缓冲区累积了足够的数据时，进行识别
        if len(accumulated_audio) >= BUFFER_SIZE:
            logger.info(f"缓冲区已满 ({len(accumulated_audio)} bytes)，开始识别...")

            try:
                # 将字节流转换为浮点数格式
                audio_float = convert_audio_to_float(accumulated_audio)

                # 调用模型进行识别
                res = model.generate(
                    input=audio_float,
                    cache={},
                    is_final=True,
                    language="zh_cn",
                    use_itn=True,
                )

                # 处理并发送识别结果
                if res and isinstance(res, list) and len(res) > 0:
                    text = res[0].get("text", "").strip()
                    if text and text != last_text:
                        logger.info(f"识别结果: {text}")

                        # 合并文本
                        full_text = merge_text(text, full_text)

                        # 将结果放入队列，供Web界面使用
                        text_queue.put({
                            'text': text,  # 当前识别的文本
                            'full_text': full_text,  # 完整的合并文本
                            'timestamp': time.time(),
                            'is_incremental': True
                        })
                        last_text = text
                    elif not text:
                        logger.info("识别结果: [无语音内容]")

            except Exception as e:
                logger.error(f"识别过程中出错: {e}")

            # 清空/滑动窗口缓冲区
            if len(accumulated_audio) > KEEP_SIZE:
                accumulated_audio = accumulated_audio[-KEEP_SIZE:]
            else:
                accumulated_audio = bytes()

    # 清理资源
    if global_process and global_process.poll() is None:
        global_process.terminate()
        try:
            global_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            global_process.kill()
    logger.info("语音识别线程退出。")


def start_recognition():
    """开始语音识别"""
    global recognition_active, full_text
    if not recognition_active:
        # 重置完整文本
        full_text = ""
        recognition_active = True
        thread = threading.Thread(target=recognition_thread)
        thread.daemon = True
        thread.start()
        logger.info("语音识别已启动")
        return True
    return False


def stop_recognition():
    """停止语音识别"""
    global recognition_active
    recognition_active = False
    logger.info("语音识别正在停止...")


def clear_text():
    """清空文本"""
    global full_text
    full_text = ""
    # 发送清空事件到队列
    text_queue.put({
        'text': '',
        'full_text': '',
        'timestamp': time.time(),
        'is_clear': True
    })


# Flask路由
@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/stream')
def stream():
    """SSE流式传输识别结果"""

    def generate():
        while True:
            try:
                # 从队列中获取识别结果，超时时间为1秒
                result = text_queue.get(timeout=1)
                if result:
                    # 格式化SSE数据
                    data = f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                    yield data
            except queue.Empty:
                # 发送心跳保持连接
                yield "data: {\"type\": \"heartbeat\"}\n\n"
            except Exception as e:
                logger.error(f"流生成错误: {e}")
                break

    return Response(generate(), mimetype="text/event-stream")


@app.route('/start')
def start_asr():
    """开始语音识别"""
    if start_recognition():
        return {"status": "success", "message": "语音识别已启动"}
    else:
        return {"status": "error", "message": "语音识别已在运行中"}


@app.route('/stop')
def stop_asr():
    """停止语音识别"""
    stop_recognition()
    return {"status": "success", "message": "语音识别已停止"}


@app.route('/clear')
def clear_asr():
    """清空文本"""
    clear_text()
    return {"status": "success", "message": "文本已清空"}


@app.route('/status')
def status():
    """获取识别状态"""
    return {
        "status": "active" if recognition_active else "inactive",
        "queue_size": text_queue.qsize(),
        "full_text_length": len(full_text)
    }


# 初始化
if __name__ == '__main__':
    if init_model():
        logger.info("启动Flask服务器...")
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    else:
        logger.error("模型初始化失败，程序退出。")
        sys.exit(1)