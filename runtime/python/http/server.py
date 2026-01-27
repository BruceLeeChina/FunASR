import argparse
import asyncio
import logging
import os
import sqlite3
import time
import uuid
from enum import Enum
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import aiofiles
import aiohttp
import ffmpeg
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modelscope.utils.logger import get_logger

from funasr import AutoModel

logger = get_logger(log_level=logging.INFO)
logger.setLevel(logging.INFO)

import os

# 在现有代码的基础上添加环境变量读取逻辑
asr_model = os.environ.get("ASR_MODEL", "paraformer-zh")
vad_model = os.environ.get("VAD_MODEL", "fsmn-vad")
punc_model = os.environ.get("PUNC_MODEL", "ct-punc-c")
device = os.environ.get("DEVICE", "cuda")
ngpu = int(os.environ.get("NGPU", "1"))
ncpu = int(os.environ.get("NCPU", "4"))
asr_model_revision = os.environ.get("ASR_MODEL_REVISION", "v2.0.4")
vad_model_revision = os.environ.get("VAD_MODEL_REVISION", "v2.0.4")
punc_model_revision = os.environ.get("PUNC_MODEL_REVISION", "v2.0.4")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=8000, required=False, help="server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default=asr_model,  # 使用环境变量的值
    help="asr model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--asr_model_revision", type=str, default=asr_model_revision, help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default=vad_model,  # 使用环境变量的值
    help="vad model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--vad_model_revision", type=str, default=vad_model_revision, help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default=punc_model,  # 使用环境变量的值
    help="model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--punc_model_revision", type=str, default=punc_model_revision, help="")
parser.add_argument("--ngpu", type=int, default=ngpu, help="0 for cpu, 1 for gpu")  # 使用环境变量的值
parser.add_argument("--device", type=str, default=device, help="cuda, cpu")  # 使用环境变量的值
parser.add_argument("--ncpu", type=int, default=ncpu, help="cpu cores")  # 使用环境变量的值

parser.add_argument(
    "--hotword_path",
    type=str,
    default="hotwords.txt",
    help="hot word txt path, only the hot word model works",
)
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")
parser.add_argument("--temp_dir", type=str, default="temp_dir/", required=False, help="temp dir")
parser.add_argument("--max_concurrent_tasks", type=int, default=10, help="Maximum number of concurrent tasks")
parser.add_argument("--db_pool_size", type=int, default=10, help="Database connection pool size")
parser.add_argument("--asr_thread_pool_size", type=int, default=4, help="ASR processing thread pool size")
args = parser.parse_args()
logger.info("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    logger.info("%s: %s" % (arg, value))
logger.info("------------------------------------------------")

os.makedirs(args.temp_dir, exist_ok=True)

logger.info("model loading")
# load funasr model
model = AutoModel(
    model=args.asr_model,
    model_revision=args.asr_model_revision,
    vad_model=args.vad_model,
    vad_model_revision=args.vad_model_revision,
    punc_model=args.punc_model,
    punc_model_revision=args.punc_model_revision,
    spk_model="cam++",
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)
logger.info("loaded models!")

app = FastAPI(title="FunASR")

# 配置模板目录
templates = Jinja2Templates(directory="templates")

# 挂载静态文件目录，提供wav文件的访问
app.mount("/data", StaticFiles(directory="data"), name="data")

param_dict = {"sentence_timestamp": True, "batch_size_s": 300}
# 会议识别参数配置
meeting_param_dict = {
    "sentence_timestamp": True,
    "batch_size_s": 300,
    "frontend": "fused_vad"  # 启用VAD功能
}

if args.hotword_path is not None and os.path.exists(args.hotword_path):
    with open(args.hotword_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    hotword = " ".join(lines)
    logger.info(f"热词：{hotword}")
    param_dict["hotword"] = hotword
    meeting_param_dict["hotword"] = hotword


# 任务状态枚举
class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# 数据库连接池
class DatabaseConnectionPool:
    def __init__(self, db_path=":memory:", pool_size=10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.lock = asyncio.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.connections.append(conn)
    
    async def get_connection(self):
        async with self.lock:
            if self.connections:
                return self.connections.pop()
            else:
                # 如果连接池为空，创建新连接（作为后备方案）
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                return conn
    
    async def return_connection(self, conn):
        async with self.lock:
            if len(self.connections) < self.pool_size:
                self.connections.append(conn)
            else:
                conn.close()
    
    async def execute(self, query, params=()):
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        finally:
            await self.return_connection(conn)
    
    async def executemany(self, query, params_list):
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor
        finally:
            await self.return_connection(conn)
    
    async def fetchone(self, query, params=()):
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            await self.return_connection(conn)
    
    async def fetchall(self, query, params=()):
        conn = await self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            await self.return_connection(conn)


# 初始化数据库连接池
db_pool = DatabaseConnectionPool(":memory:", args.db_pool_size)

# 创建任务表
async def init_db():
    await db_pool.execute('''
    CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY,
        task_type TEXT,
        file_path TEXT,
        file_url TEXT,
        file_name TEXT,
        status TEXT,
        progress REAL,
        result TEXT,
        error_msg TEXT,
        created_time INTEGER,
        updated_time INTEGER,
        callback_url TEXT,
        callback_status TEXT,
        app_id TEXT,
        biz_type TEXT,
        biz_unique_id TEXT UNIQUE,
        recognition_mode TEXT DEFAULT 'default'
    )
    ''')

# 任务队列和并发控制
task_queue = asyncio.Queue()
running_tasks = set()
asr_thread_pool = ThreadPoolExecutor(max_workers=args.asr_thread_pool_size)

# 日志配置，增加更多调试信息
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def download_file(url: str, save_path: str) -> bool:
    """下载文件"""
    try:
        # 创建不验证SSL证书的会话
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(await response.read())
                    return True
                else:
                    logger.error(f"下载文件失败，状态码: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"下载文件时发生错误: {e}")
        return False


async def send_callback_notification(task_id: str, callback_url: str) -> bool:
    """发送回调通知"""
    try:
        # 获取任务详情
        task_result = await db_pool.fetchone(
            "SELECT task_id, status, result, error_msg, app_id, biz_type, biz_unique_id FROM tasks WHERE task_id = ?",
            (task_id,)
        )

        if not task_result:
            logger.error(f"无法找到任务{task_id}的信息，无法发送回调")
            return False

        task_id, status, result_data, error_msg, app_id, biz_type, biz_unique_id = task_result

        # 构建回调数据
        callback_data = {
            "task_id": task_id,
            "status": status,
            "timestamp": int(time.time())
        }

        # 添加业务标识信息
        if app_id:
            callback_data["app_id"] = app_id
        if biz_type:
            callback_data["biz_type"] = biz_type
        if biz_unique_id:
            callback_data["biz_unique_id"] = biz_unique_id

        import json
        if status == TaskStatus.COMPLETED.value and result_data:
            callback_data["result"] = json.loads(result_data)
        elif status == TaskStatus.FAILED.value and error_msg:
            callback_data["error_msg"] = error_msg

        # 创建不验证SSL证书的会话
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(callback_url, json=callback_data, timeout=10) as response:
                if response.status == 200:
                    # 更新回调状态为成功
                    await db_pool.execute(
                        "UPDATE tasks SET callback_status = ? WHERE task_id = ?",
                        ("success", task_id)
                    )
                    logger.info(f"任务{task_id}的回调通知发送成功")
                    return True
                else:
                    logger.error(f"任务{task_id}的回调通知发送失败，状态码: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"发送回调通知时发生错误: {e}")
        return False


# 全局会议识别模型实例，避免重复初始化
meeting_model = None

def init_meeting_model():
    """
    初始化会议识别模型
    """
    global meeting_model
    if meeting_model is None:
        print("正在初始化会议识别模型...")
        meeting_model = AutoModel(
            model="paraformer-zh",  # ASR模型
            vad_model="fsmn-vad",  # 语音活动检测模型
            punc_model="ct-punc",  # 标点符号模型
            spk_model="cam++"  # 说话人识别模型
        )
        print("会议识别模型初始化完成")

def process_meeting_audio(audio_path):
    """
    处理会议音频并返回对话格式结果
    """
    global meeting_model
    print(f"正在处理会议音频: {audio_path}")
    
    # 初始化模型（如果尚未初始化）
    if meeting_model is None:
        init_meeting_model()

    print("开始识别会议音频...")
    # 生成识别结果
    result = meeting_model.generate(
        input=audio_path,
        batch_size_s=300,
        hotword=""
    )

    print(f"模型返回原始结果: {result}")  # 添加调试输出

    # 解析结果并格式化为对话
    formatted_dialogue = []
    if result and len(result) > 0:
        result_item = result[0]
        
        # 检查是否包含sentence_info（包含说话人信息）
        if isinstance(result_item, dict) and 'sentence_info' in result_item:
            sentences = result_item['sentence_info']
            print(f"检测到 {len(sentences)} 个带说话人信息的语音段")
            
            for i, seg in enumerate(sentences):
                print(f"处理第 {i+1} 个片段: {seg}")  # 调试信息
                dialogue_entry = {
                    'speaker': f'Speaker {seg.get("spk", "Unknown")}',
                    'text': seg.get('text', ''),
                    'start_time': seg.get('start', 0) / 1000.0,  # 转换为秒
                    'end_time': seg.get('end', 0) / 1000.0  # 转换为秒
                }
                formatted_dialogue.append(dialogue_entry)
        else:
            # 如果没有sentence_info，尝试其他结构
            print("未找到sentence_info，尝试其他结构...")
            # 可能是整体文本，尝试按时间戳分割
            if isinstance(result_item, dict) and 'timestamp' in result_item and 'text' in result_item:
                text = result_item['text']
                timestamps = result_item['timestamp']
                
                # 简单按时间戳分割
                for i, ts in enumerate(timestamps):
                    if len(ts) >= 2:
                        start_time, end_time = ts
                        dialogue_entry = {
                            'speaker': 'Speaker Unknown',
                            'text': text,  # 这里无法准确分割文本
                            'start_time': start_time / 1000.0,
                            'end_time': end_time / 1000.0
                        }
                        formatted_dialogue.append(dialogue_entry)

    return formatted_dialogue





async def process_audio_file(audio_path: str, is_meeting_mode: bool = False) -> Dict[str, Any]:
    """处理音频文件并进行识别"""
    loop = asyncio.get_event_loop()
    # 使用线程池执行CPU密集型的ASR任务
    func = partial(_process_audio_sync, audio_path, is_meeting_mode)
    result = await loop.run_in_executor(asr_thread_pool, func)
    return result

def _process_audio_sync(audio_path: str, is_meeting_mode: bool = False) -> Dict[str, Any]:
    """在单独线程中执行的实际音频处理函数"""
    try:
        if is_meeting_mode:
            # 会议识别模式 - 直接在同步函数中处理
            formatted_dialogue = process_meeting_audio(audio_path)
            return {
                "dialogue": formatted_dialogue,
                "code": 0
            }
        else:
            # 使用ffmpeg转换音频格式
            audio_bytes, _ = (
                ffmpeg.input(audio_path, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )

            # 进行语音识别
            rec_results = model.generate(input=audio_bytes, is_final=True, **param_dict)

            # 解析识别结果
            if len(rec_results) > 0 and "text" in rec_results[0]:
                rec_result = rec_results[0]
                text = rec_result["text"]
                sentences = []
                if "sentence_info" in rec_result:
                    for sentence in rec_result["sentence_info"]:
                        sentences.append({
                            "text": sentence["text"],
                            "start": sentence["start"],
                            "end": sentence["end"]
                        })

                return {
                    "text": text,
                    "sentences": sentences,
                    "code": 0
                }
            else:
                return {
                    "text": "",
                    "sentences": [],
                    "code": 0
                }
    except Exception as e:
        logger.error(f"处理音频文件时发生错误: {e}")
        raise


async def task_processor():
    """任务处理器"""
    global task_queue
    logger.info("任务处理器已启动并开始监听任务队列")

    while True:
        try:
            logger.debug(f"任务处理器等待新任务，队列大小: {task_queue.qsize()}")
            task_info = await task_queue.get()
            logger.debug(f"从队列获取到任务信息: {task_info}")

            # 从任务信息中提取task_id
            task_id = task_info["task_id"]
            logger.info(f"开始处理任务: {task_id}")

            # 检查任务是否已取消
            status_result = await db_pool.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))
            if status_result and status_result[0] == TaskStatus.CANCELED.value:
                logger.info(f"任务{task_id}已取消，跳过处理")
                task_queue.task_done()
                continue

            # 检查并发任务数
            if len(running_tasks) >= args.max_concurrent_tasks:
                logger.debug(
                    f"当前并发任务数已达上限: {len(running_tasks)}/{args.max_concurrent_tasks}，将任务{task_id}重新放回队列")
                await asyncio.sleep(0.1)
                await task_queue.put(task_info)
                logger.debug(f"任务{task_id}已重新放回队列，当前队列大小: {task_queue.qsize()}")
                continue

            running_tasks.add(task_id)
            logger.debug(f"任务{task_id}已添加到运行任务集合，当前运行任务数: {len(running_tasks)}")

            try:
                # 更新任务状态为处理中
                logger.debug(f"开始处理任务: {task_id}")
                await db_pool.execute(
                    "UPDATE tasks SET status = ?, progress = 0.1, updated_time = ? WHERE task_id = ?",
                    (TaskStatus.PROCESSING.value, int(time.time()), task_id)
                )
                logger.info(f"任务{task_id}状态已更新为处理中")

                # 再次检查任务是否已取消
                status_result = await db_pool.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))
                if status_result and status_result[0] == TaskStatus.CANCELED.value:
                    logger.info(f"任务{task_id}已取消，停止处理")
                    await db_pool.execute(
                        "UPDATE tasks SET progress = 0, updated_time = ? WHERE task_id = ?",
                        (int(time.time()), task_id)
                    )
                    continue

                # 获取任务详细信息
                db_task_info = await db_pool.fetchone("SELECT task_type, file_path, file_url, file_name, recognition_mode FROM tasks WHERE task_id = ?",
                               (task_id,))

                if not db_task_info:
                    logger.error(f"任务{task_id}信息不存在于数据库中")
                    raise Exception("任务信息不存在")

                task_type, file_path, file_url, file_name, recognition_mode = db_task_info
                audio_path = file_path
                logger.debug(
                    f"任务{task_id}信息: task_type={task_type}, file_path={file_path}, file_url={file_url}, file_name={file_name}, recognition_mode={recognition_mode}")

                # 根据任务类型处理
                if task_type == "file_url" and file_url:
                    # 下载文件
                    await db_pool.execute("UPDATE tasks SET progress = 0.2, updated_time = ? WHERE task_id = ?",
                                   (int(time.time()), task_id))

                    file_ext = os.path.splitext(file_name)[1] if file_name else ".wav"
                    temp_file_path = f"{args.temp_dir}/{task_id}{file_ext}"

                    if not await download_file(file_url, temp_file_path):
                        raise Exception("文件下载失败")

                    audio_path = temp_file_path
                    await db_pool.execute("UPDATE tasks SET file_path = ?, progress = 0.4, updated_time = ? WHERE task_id = ?",
                                   (audio_path, int(time.time()), task_id))

                # 处理音频文件
                await db_pool.execute("UPDATE tasks SET progress = 0.6, updated_time = ? WHERE task_id = ?",
                               (int(time.time()), task_id))
                logger.info(f"任务{task_id}开始进行语音识别")

                # 判断是否为会议识别模式
                is_meeting_mode = recognition_mode == "meeting"
                result = await process_audio_file(audio_path, is_meeting_mode=is_meeting_mode)
                logger.debug(f"任务{task_id}语音识别完成，结果: {result}")

                # 将结果转换为标准JSON字符串存储
                import json
                result_json = json.dumps(result)
                # 更新任务状态为已完成
                await db_pool.execute(
                    "UPDATE tasks SET status = ?, progress = 1.0, result = ?, updated_time = ? WHERE task_id = ?",
                    (TaskStatus.COMPLETED.value, result_json, int(time.time()), task_id)
                )
                logger.info(f"任务{task_id}处理完成")

                # 检查是否有回调URL，如果有则发送回调通知
                callback_url_result = await db_pool.fetchone("SELECT callback_url FROM tasks WHERE task_id = ?", (task_id,))
                if callback_url_result and callback_url_result[0]:
                    callback_url = callback_url_result[0]
                    # 在后台发送回调，不阻塞任务处理
                    asyncio.create_task(send_callback_notification(task_id, callback_url))

            except Exception as e:
                # 更新任务状态为失败
                logger.error(f"任务{task_id}处理失败: {e}")
                await db_pool.execute(
                    "UPDATE tasks SET status = ?, progress = 0, error_msg = ?, updated_time = ? WHERE task_id = ?",
                    (TaskStatus.FAILED.value, str(e), int(time.time()), task_id)
                )
                logger.info(f"任务{task_id}状态已更新为失败")

                # 检查是否有回调URL，如果有则发送回调通知
                callback_url_result = await db_pool.fetchone("SELECT callback_url FROM tasks WHERE task_id = ?", (task_id,))
                if callback_url_result and callback_url_result[0]:
                    callback_url = callback_url_result[0]
                    # 在后台发送回调，不阻塞任务处理
                    asyncio.create_task(send_callback_notification(task_id, callback_url))
            finally:
                running_tasks.remove(task_id)
                logger.debug(f"任务{task_id}已从运行任务集合中移除，当前运行任务数: {len(running_tasks)}")
                task_queue.task_done()
                logger.debug(f"任务{task_id}已标记为完成，当前队列大小: {task_queue.qsize()}")
        except Exception as e:
            logger.error(f"任务处理器内部错误: {e}")
            # 确保队列任务已标记完成
            try:
                if 'task_id' in locals():
                    task_queue.task_done()
            except:
                pass


# 启动任务处理器
@app.on_event("startup")
async def startup_event():
    global task_queue
    # 初始化数据库
    await init_db()
    
    # 在FastAPI的事件循环中初始化任务队列
    task_queue = asyncio.Queue()
    logger.debug(f"任务队列已初始化: {task_queue}")
    logger.debug(f"当前事件循环: {asyncio.get_running_loop()}")
    logger.info(f"任务处理器已启动")
    # 创建多个任务处理器以支持并发处理
    for i in range(min(args.asr_thread_pool_size, args.max_concurrent_tasks)):
        processor_task = asyncio.create_task(task_processor())
        logger.debug(f"任务处理器 {i+1} 已创建: {processor_task}")


@app.post("/mock_callback")
async def mock_callback(request: Request):
    """模拟回调接收接口，用于测试回调功能"""
    try:
        callback_data = await request.json()
        logger.info(f"收到回调通知: {callback_data}")
        return {
            "code": 0,
            "msg": "回调接收成功",
            "received_data": callback_data
        }
    except Exception as e:
        logger.error(f"处理回调请求时发生错误: {e}")
        return {
            "code": 1,
            "msg": f"回调处理失败: {str(e)}"
        }


@app.post("/submit_task")
async def submit_task(
        file: Optional[UploadFile] = File(None, description="音频文件"),
        file_url: Optional[str] = Form(None, description="音频文件URL"),
        file_name: Optional[str] = Form(None, description="文件名"),
        callback_url: Optional[str] = Form(None, description="任务完成后回调URL"),
        app_id: Optional[str] = Form(None, description="应用ID"),
        biz_type: Optional[str] = Form(None, description="业务类型"),
        biz_unique_id: Optional[str] = Form(None, description="业务唯一ID"),
        recognition_mode: Optional[str] = Form("default", description="识别模式: default 或 meeting")
):
    """提交识别任务"""
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="必须提供音频文件或文件URL")

    task_id = str(uuid.uuid1())
    current_time = int(time.time())
    task_type = "file_upload" if file else "file_url"
    file_path = ""

    if file:
        # 保存上传的文件
        suffix = file.filename.split(".")[-1] if file.filename else "wav"
        file_path = f"{args.temp_dir}/{task_id}.{suffix}"
        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        file_name = file.filename or f"upload_{task_id}.{suffix}"

    # 插入任务记录
    await db_pool.execute(
        "INSERT INTO tasks (task_id, task_type, file_path, file_url, file_name, status, progress, result, error_msg, created_time, updated_time, callback_url, callback_status, app_id, biz_type, biz_unique_id, recognition_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, task_type, file_path, file_url, file_name, TaskStatus.PENDING.value, 0, "", "", current_time,
         current_time, callback_url, "pending", app_id, biz_type, biz_unique_id, recognition_mode)
    )

    # 将任务加入队列
    global task_queue
    task_info = {
        "task_id": task_id,
        "task_type": task_type,
        "file_path": file_path,
        "file_url": file_url
    }
    await task_queue.put(task_info)
    logger.info(f"任务{task_id}已添加到处理队列")
    logger.debug(f"队列中任务数量: {task_queue.qsize()}")

    return {
        "code": 0,
        "msg": "任务提交成功",
        "task_id": task_id
    }


@app.get("/get_task_status")
async def get_task_status(task_id: str = Query(..., description="任务ID")):
    """查询任务状态"""
    result = await db_pool.fetchone("SELECT status, progress, updated_time, callback_status FROM tasks WHERE task_id = ?", (task_id,))

    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")

    status, progress, updated_time, callback_status = result

    return {
        "code": 0,
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "updated_time": updated_time,
        "callback_status": callback_status
    }


@app.get("/get_task_result")
async def get_task_result(task_id: str = Query(..., description="任务ID")):
    """查询任务结果"""
    result = await db_pool.fetchone("SELECT status, result, error_msg, callback_status FROM tasks WHERE task_id = ?", (task_id,))

    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")

    status, result_data, error_msg, callback_status = result

    if status == TaskStatus.PENDING.value or status == TaskStatus.PROCESSING.value:
        return {
            "code": 1,
            "msg": "任务尚未完成",
            "status": status
        }
    elif status == TaskStatus.COMPLETED.value:
        import json
        try:
            result_json = json.loads(result_data) if result_data else {}
            return {
                "code": 0,
                "status": status,
                "result": result_json,
                "callback_status": callback_status
            }
        except json.JSONDecodeError as e:
            return {
                "code": 4,
                "status": status,
                "error_msg": f"结果解析失败: {str(e)}"
            }
    elif status == TaskStatus.FAILED.value:
        return {
            "code": 2,
            "status": status,
            "error_msg": error_msg,
            "callback_status": callback_status
        }
    elif status == TaskStatus.CANCELED.value:
        return {
            "code": 3,
            "status": status,
            "msg": "任务已取消",
            "callback_status": callback_status
        }


@app.post("/cancel_task")
async def cancel_task(task_id: str = Form(..., description="任务ID")):
    """取消任务"""
    result = await db_pool.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))

    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")

    status = result[0]

    if status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELED.value]:
        return {
            "code": 1,
            "msg": "任务已完成或失败或已取消，无法取消"
        }

    # 更新任务状态为已取消
    await db_pool.execute(
        "UPDATE tasks SET status = ?, updated_time = ? WHERE task_id = ?",
        (TaskStatus.CANCELED.value, int(time.time()), task_id)
    )

    return {
        "code": 0,
        "msg": "任务取消成功"
    }


@app.post("/delete_task")
async def delete_task(task_id: str = Form(..., description="任务ID")):
    """删除任务"""
    result = await db_pool.fetchone("SELECT file_path FROM tasks WHERE task_id = ?", (task_id,))

    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 删除本地文件
    file_path = result[0]
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"删除文件失败: {e}")

    # 删除任务记录
    await db_pool.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    return {
        "code": 0,
        "msg": "任务删除成功"
    }


@app.get("/list_tasks")
async def list_tasks(page: int = Query(1, ge=1, description="页码"),
                     page_size: int = Query(10, ge=1, le=100, description="每页数量"),
                     status: Optional[str] = Query(None, description="任务状态过滤"),
                     task_type: Optional[str] = Query(None, description="任务类型过滤"),
                     recognition_mode: Optional[str] = Query(None, description="识别模式过滤")):
    """查询任务列表"""
    try:
        # 将空字符串转换为None
        if status == "":
            status = None
        if task_type == "":
            task_type = None
        if recognition_mode == "":
            recognition_mode = None
            
        offset = (page - 1) * page_size
        # 构建查询条件
        conditions = []
        params = []
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if recognition_mode:
            conditions.append("recognition_mode = ?")
            params.append(recognition_mode)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        tasks = await db_pool.fetchall(
            f"SELECT task_id, task_type, file_name, status, progress, created_time, updated_time, callback_status, biz_type, biz_unique_id, recognition_mode FROM tasks {where_clause} ORDER BY created_time DESC LIMIT ? OFFSET ?",
            (*params, page_size, offset)
        )

        total_row = await db_pool.fetchone(f"SELECT COUNT(*) FROM tasks {where_clause}", params)

        total = total_row[0] if total_row is not None else 0

        task_list = []
        for task in tasks:
            task_list.append({
                "task_id": task[0],
                "task_type": task[1],
                "file_name": task[2],
                "status": task[3],
                "progress": task[4],
                "created_time": task[5],
                "updated_time": task[6],
                "callback_status": task[7],
                "biz_type": task[8],
                "biz_unique_id": task[9],
                "recognition_mode": task[10]
            })

        return {"code": 0, "msg": "查询任务列表成功", "tasks": task_list, "total": total, "page": page,
                "limit": page_size}
    except Exception as e:
        return {"code": 1, "msg": f"查询任务列表失败: {str(e)}"}



@app.get("/get_task_details")
async def get_task_details(task_id: str = Query(..., description="任务ID")):
    """查询任务详情"""
    try:
        task = await db_pool.fetchone(
            "SELECT task_id, task_type, file_path, file_url, file_name, status, progress, result, error_msg, created_time, updated_time, callback_status, recognition_mode FROM tasks WHERE task_id = ?",
            (task_id,)
        )

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        import json
        task_info = {
            "task_id": task[0],
            "task_type": task[1],
            "file_path": task[2],
            "file_url": task[3],
            "file_name": task[4],
            "status": task[5],
            "progress": task[6],
            "result": json.loads(task[7]) if task[7] else {},
            "error_message": task[8],
            "created_time": task[9],
            "updated_time": task[10],
            "callback_status": task[11],
            "recognition_mode": task[12]
        }

        return {"code": 0, "msg": "查询任务详情成功", "task": task_info}
    except Exception as e:
        return {"code": 1, "msg": f"查询任务详情失败: {str(e)}"}


@app.post("/batch_get_task_status")
async def batch_get_task_status(task_ids: str = Form(..., description="任务ID列表，逗号分隔")):
    """批量查询任务状态"""
    if not task_ids:
        raise HTTPException(status_code=400, detail="必须提供任务ID列表")

    # 解析任务ID列表
    task_list = [tid.strip() for tid in task_ids.split(",") if tid.strip()]
    if not task_list:
        raise HTTPException(status_code=400, detail="任务ID列表不能为空")

    # 批量查询任务状态
    placeholders = ",".join(["?"] * len(task_list))
    results = await db_pool.fetchall(
        f"SELECT task_id, status, progress, updated_time, callback_status FROM tasks WHERE task_id IN ({placeholders})",
        task_list
    )

    # 构建结果字典
    task_status_map = {}
    for result in results:
        task_id, status, progress, updated_time, callback_status = result
        task_status_map[task_id] = {
            "code": 0,
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "updated_time": updated_time,
            "callback_status": callback_status
        }

    # 为未找到的任务添加错误信息
    for task_id in task_list:
        if task_id not in task_status_map:
            task_status_map[task_id] = {
                "code": 1,
                "task_id": task_id,
                "msg": "任务不存在"
            }

    return {
        "code": 0,
        "msg": "批量查询任务状态完成",
        "results": list(task_status_map.values())
    }


@app.post("/batch_get_task_result")
async def batch_get_task_result(task_ids: str = Form(..., description="任务ID列表，逗号分隔")):
    """批量查询任务结果"""
    if not task_ids:
        raise HTTPException(status_code=400, detail="必须提供任务ID列表")

    # 解析任务ID列表
    task_list = [tid.strip() for tid in task_ids.split(",") if tid.strip()]
    if not task_list:
        raise HTTPException(status_code=400, detail="任务ID列表不能为空")

    # 批量查询任务结果
    placeholders = ",".join(["?"] * len(task_list))
    results = await db_pool.fetchall(
        f"SELECT task_id, status, result, error_msg, callback_status FROM tasks WHERE task_id IN ({placeholders})",
        task_list
    )

    # 构建结果字典
    task_result_map = {}
    import json
    for result in results:
        task_id, status, result_data, error_msg, callback_status = result

        if status == TaskStatus.PENDING.value or status == TaskStatus.PROCESSING.value:
            task_result_map[task_id] = {
                "code": 1,
                "task_id": task_id,
                "status": status,
                "msg": "任务尚未完成",
                "callback_status": callback_status
            }
        elif status == TaskStatus.COMPLETED.value:
            try:
                result_json = json.loads(result_data) if result_data else {}
                task_result_map[task_id] = {
                    "code": 0,
                    "task_id": task_id,
                    "status": status,
                    "result": result_json,
                    "callback_status": callback_status
                }
            except json.JSONDecodeError as e:
                task_result_map[task_id] = {
                    "code": 4,
                    "task_id": task_id,
                    "status": status,
                    "error_msg": f"结果解析失败: {str(e)}",
                    "callback_status": callback_status
                }
        elif status == TaskStatus.FAILED.value:
            task_result_map[task_id] = {
                "code": 2,
                "task_id": task_id,
                "status": status,
                "error_msg": error_msg,
                "callback_status": callback_status
            }
        elif status == TaskStatus.CANCELED.value:
            task_result_map[task_id] = {
                "code": 3,
                "task_id": task_id,
                "status": status,
                "msg": "任务已取消",
                "callback_status": callback_status
            }

    # 为未找到的任务添加错误信息
    for task_id in task_list:
        if task_id not in task_result_map:
            task_result_map[task_id] = {
                "code": 1,
                "task_id": task_id,
                "msg": "任务不存在"
            }

    return {
        "code": 0,
        "msg": "批量查询任务结果完成",
        "results": list(task_result_map.values())
    }


@app.post("/batch_get_task_details")
async def batch_get_task_details(task_ids: str = Form(..., description="任务ID列表，逗号分隔")):
    """批量查询任务详情"""
    if not task_ids:
        raise HTTPException(status_code=400, detail="必须提供任务ID列表")

    # 解析任务ID列表
    task_list = [tid.strip() for tid in task_ids.split(",") if tid.strip()]
    if not task_list:
        raise HTTPException(status_code=400, detail="任务ID列表不能为空")

    # 批量查询任务详情
    placeholders = ",".join(["?"] * len(task_list))
    results = await db_pool.fetchall(
        f"SELECT task_id, task_type, file_path, file_url, file_name, status, progress, result, error_msg, created_time, updated_time, recognition_mode FROM tasks WHERE task_id IN ({placeholders})",
        task_list
    )

    # 构建结果字典
    task_details_map = {}
    import json
    for result in results:
        task_id, task_type, file_path, file_url, file_name, status, progress, result_data, error_msg, created_time, updated_time = result

        task_details = {
            "task_id": task_id,
            "task_type": task_type,
            "file_path": file_path,
            "file_url": file_url,
            "file_name": file_name,
            "status": status,
            "progress": progress,
            "error_msg": error_msg,
            "created_time": created_time,
            "updated_time": updated_time,
            "recognition_mode": recognition_mode
        }

        if status == TaskStatus.COMPLETED.value and result_data:
            try:
                task_details["result"] = json.loads(result_data)
            except json.JSONDecodeError:
                pass

        task_details_map[task_id] = {
            "code": 0,
            "task_details": task_details
        }

    # 为未找到的任务添加错误信息
    for task_id in task_list:
        if task_id not in task_details_map:
            task_details_map[task_id] = {
                "code": 1,
                "task_id": task_id,
                "msg": "任务不存在"
            }

    return {
        "code": 0,
        "msg": "批量查询任务详情完成",
        "results": list(task_details_map.values())
    }


@app.post("/batch_operation")
async def batch_operation(
        operation: str = Form(..., description="操作类型: submit/cancel/delete"),
        task_ids: Optional[str] = Form(None, description="任务ID列表，逗号分隔"),
        file_urls: Optional[str] = Form(None, description="文件URL列表，逗号分隔"),
        file_names: Optional[str] = Form(None, description="文件名列表，逗号分隔"),
        callback_url: Optional[str] = Form(None, description="任务完成后回调URL"),
        file_list: Optional[str] = Form(None,
                                        description='文件列表，JSON格式: [{"file_url":"xxx","file_name":"xxx","callback_url":"xx"}]'),
        app_id: Optional[str] = Form(None, description="应用ID"),
        biz_type: Optional[str] = Form(None, description="业务类型"),
        biz_unique_id: Optional[str] = Form(None, description="业务唯一ID"),
        recognition_mode: Optional[str] = Form("default", description="识别模式: default 或 meeting")
):
    """批量操作"""
    results = []

    if operation == "submit":
        # 批量提交任务
        file_items = []

        # 如果提供了file_list JSON参数，则优先使用它
        if file_list:
            try:
                import json
                file_items = json.loads(file_list)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"file_list参数不是有效的JSON格式: {str(e)}")
        elif file_urls:
            # 兼容旧的方式
            urls = file_urls.split(",")
            names = file_names.split(",") if file_names else [f"batch_{i}.wav" for i in range(len(urls))]
            callbacks = [callback_url] * len(urls) if callback_url else [None] * len(urls)

            file_items = [
                {"file_url": url, "file_name": name, "callback_url": cb_url}
                for url, name, cb_url in zip(urls, names, callbacks)
            ]
        else:
            raise HTTPException(status_code=400, detail="批量提交任务必须提供文件列表")

        for item in file_items:
            try:
                file_url = item.get("file_url")
                file_name = item.get("file_name")
                item_callback_url = item.get("callback_url") or callback_url
                item_app_id = item.get("app_id") or app_id
                item_biz_type = item.get("biz_type") or biz_type
                item_biz_unique_id = item.get("biz_unique_id") or biz_unique_id

                if not file_url:
                    results.append({
                        "code": 1,
                        "msg": "缺少file_url参数",
                        "item": item
                    })
                    continue

                # 调用submit_task逻辑
                task_id = str(uuid.uuid1())
                current_time = int(time.time())

                # 插入任务记录
                await db_pool.execute(
                    "INSERT INTO tasks (task_id, task_type, file_path, file_url, file_name, status, progress, result, error_msg, created_time, updated_time, callback_url, callback_status, app_id, biz_type, biz_unique_id, recognition_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                    task_id, "file_url", "", file_url, file_name or f"batch_{task_id}.wav", TaskStatus.PENDING.value, 0,
                    "", "", current_time, current_time, item_callback_url, "pending", item_app_id, item_biz_type,
                    item_biz_unique_id, recognition_mode)
                )

                # 将任务加入队列
                logger.debug(f"当前task_queue: {task_queue}")
                logger.debug(f"当前事件循环: {asyncio.get_running_loop()}")
                logger.debug(f"准备将批量任务{task_id}添加到队列，当前队列大小: {task_queue.qsize()}")
                task_info = {
                    "task_id": task_id,
                    "task_type": "file_url",
                    "file_path": "",
                    "file_url": file_url
                }
                await task_queue.put(task_info)
                logger.debug(f"批量任务{task_id}已成功添加到队列，当前队列大小: {task_queue.qsize()}")
                logger.info(f"批量任务已添加到队列: {task_id}")

                results.append({
                    "code": 0,
                    "msg": "任务提交成功",
                    "task_id": task_id
                })
            except Exception as e:
                results.append({
                    "code": 1,
                    "msg": f"任务提交失败: {str(e)}",
                    "item": item
                })

    elif operation in ["cancel", "delete"]:
        # 批量取消或删除任务
        if not task_ids:
            raise HTTPException(status_code=400, detail="批量操作必须提供任务ID列表")

        task_list = task_ids.split(",")

        for task_id in task_list:
            try:
                if operation == "cancel":
                    # 取消任务
                    result = await db_pool.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))

                    if not result:
                        results.append({
                            "code": 1,
                            "msg": "任务不存在",
                            "task_id": task_id
                        })
                        continue

                    status = result[0]

                    if status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELED.value]:
                        results.append({
                            "code": 1,
                            "msg": "任务已完成或失败或已取消，无法取消",
                            "task_id": task_id
                        })
                        continue

                    # 更新任务状态为已取消
                    await db_pool.execute(
                        "UPDATE tasks SET status = ?, updated_time = ? WHERE task_id = ?",
                        (TaskStatus.CANCELED.value, int(time.time()), task_id)
                    )

                    results.append({
                        "code": 0,
                        "msg": "任务取消成功",
                        "task_id": task_id
                    })

                else:  # delete
                    # 删除任务
                    result = await db_pool.fetchone("SELECT file_path FROM tasks WHERE task_id = ?", (task_id,))

                    if not result:
                        results.append({
                            "code": 1,
                            "msg": "任务不存在",
                            "task_id": task_id
                        })
                        continue

                    # 删除本地文件
                    file_path = result[0]
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.error(f"删除文件失败: {e}")

                    # 删除任务记录
                    await db_pool.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

                    results.append({
                        "code": 0,
                        "msg": "任务删除成功",
                        "task_id": task_id
                    })
            except Exception as e:
                results.append({
                    "code": 1,
                    "msg": f"操作失败: {str(e)}",
                    "task_id": task_id
                })

    else:
        raise HTTPException(status_code=400, detail="不支持的操作类型")

    return {
        "code": 0,
        "msg": "批量操作完成",
        "results": results
    }


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file")):
    """向后兼容的识别接口"""
    suffix = audio.filename.split(".")[-1] if audio.filename else "wav"
    audio_path = f"{args.temp_dir}/{str(uuid.uuid1())}.{suffix}"
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        logger.error(f"读取音频文件发生错误，错误信息：{e}")
        return {"msg": "读取音频文件发生错误", "code": 1}
    rec_results = model.generate(input=audio_bytes, is_final=True, **param_dict)
    # 结果为空
    if len(rec_results[0]["text"]) == 0:
        return {"text": "", "sentences": [], "code": 0}
    elif len(rec_results[0]["text"]) > 0:
        # 解析识别结果
        rec_result = rec_results[0]
        text = rec_result["text"]
        sentences = []
        for sentence in rec_result["sentence_info"]:
            # 每句话的时间戳
            sentences.append(
                {"text": sentence["text"], "start": sentence["start"], "end": sentence["end"]}
            )
        ret = {"text": text, "sentences": sentences, "code": 0}
        logger.info(f"识别结果：{ret}")
        return ret
    else:
        logger.info(f"识别结果：{rec_results}")
        return {"msg": "未知错误", "code": -1}


@app.get("/get_task_by_biz_id")
async def get_task_by_biz_id(
        biz_unique_id: Optional[str] = Query(None, description="业务唯一ID"),
        app_id: Optional[str] = Query(None, description="应用ID"),
        recognition_mode: Optional[str] = Query(None, description="识别模式")
):
    """根据业务唯一ID和应用ID查询任务，至少需要提供一个条件"""
    try:
        # 将空字符串转换为None
        if biz_unique_id == "":
            biz_unique_id = None
        if app_id == "":
            app_id = None
        if recognition_mode == "":
            recognition_mode = None
            
        # 验证至少提供一个查询条件
        if not biz_unique_id and not app_id:
            raise HTTPException(status_code=400, detail="请至少提供一个查询条件")
            
        # 构建查询条件
        query_conditions = []
        query_params = []
        
        # 基础查询字段
        base_query = "SELECT task_id, task_type, file_path, file_url, file_name, status, progress, result, error_msg, created_time, updated_time, callback_status, app_id, biz_type, biz_unique_id, recognition_mode FROM tasks WHERE "
        
        # 添加查询条件
        if biz_unique_id:
            query_conditions.append("biz_unique_id = ?")
            query_params.append(biz_unique_id)
        
        if app_id:
            query_conditions.append("app_id = ?")
            query_params.append(app_id)
        
        if recognition_mode:
            query_conditions.append("recognition_mode = ?")
            query_params.append(recognition_mode)
        
        # 构建完整查询语句
        query = base_query + " AND ".join(query_conditions)
        
        task = await db_pool.fetchone(query, tuple(query_params))

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        import json
        task_info = {
            "task_id": task[0],
            "task_type": task[1],
            "file_path": task[2],
            "file_url": task[3],
            "file_name": task[4],
            "status": task[5],
            "progress": task[6],
            "result": json.loads(task[7]) if task[7] else {},
            "error_message": task[8],
            "created_time": task[9],
            "updated_time": task[10],
            "callback_status": task[11],
            "app_id": task[12],
            "biz_type": task[13],
            "biz_unique_id": task[14],
            "recognition_mode": task[15]
        }

        return {"code": 0, "msg": "查询任务详情成功", "task": task_info}
    except Exception as e:
        return {"code": 1, "msg": f"查询任务详情失败: {str(e)}"}


@app.get("/list_tasks_by_app")
async def list_tasks_by_app(
        app_id: Optional[str] = Query(None, description="应用ID"),
        biz_type: Optional[str] = Query(None, description="业务类型"),
        recognition_mode: Optional[str] = Query(None, description="识别模式"),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(10, ge=1, le=100, description="每页数量")
):
    """根据应用ID和业务类型查询任务列表，至少需要提供一个条件"""
    try:
        # 将空字符串转换为None
        if app_id == "":
            app_id = None
        if biz_type == "":
            biz_type = None
        if recognition_mode == "":
            recognition_mode = None
            
        # 验证至少提供一个查询条件
        if not app_id and not biz_type and not recognition_mode:
            raise HTTPException(status_code=400, detail="请至少提供一个查询条件")
            
        offset = (page - 1) * page_size
        # 构建查询条件
        conditions = []
        params = []
        
        if app_id:
            conditions.append("app_id = ?")
            params.append(app_id)
        if biz_type:
            conditions.append("biz_type = ?")
            params.append(biz_type)
        if recognition_mode:
            conditions.append("recognition_mode = ?")
            params.append(recognition_mode)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params = tuple(params)

        tasks = await db_pool.fetchall(
            f"SELECT task_id, task_type, file_name, status, created_time, updated_time, callback_status, biz_type, biz_unique_id, recognition_mode FROM tasks {where_clause} ORDER BY created_time DESC LIMIT ? OFFSET ?",
            (*params, page_size, offset)
        )

        total_row = await db_pool.fetchone(f"SELECT COUNT(*) FROM tasks {where_clause}", params)

        total = total_row[0] if total_row is not None else 0

        task_list = []
        for task in tasks:
            task_list.append({
                "task_id": task[0],
                "task_type": task[1],
                "file_name": task[2],
                "status": task[3],
                "created_time": task[4],
                "updated_time": task[5],
                "callback_status": task[6],
                "biz_type": task[7],
                "biz_unique_id": task[8],
                "recognition_mode": task[9]
            })

        return {"code": 0, "msg": "查询任务列表成功", "tasks": task_list, "total": total, "page": page,
                "limit": page_size}
    except Exception as e:
        return {"code": 1, "msg": f"查询任务列表失败: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """HTML测试页面"""
    return templates.TemplateResponse("index.html", {"request": request})


import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile
    )