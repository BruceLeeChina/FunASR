#!/bin/bash

# FunASR 离线文本识别接口启动脚本

# 设置默认参数
HOST="0.0.0.0"
PORT="8000"
ASR_MODEL="paraformer-zh"
VAD_MODEL="fsmn-vad"
PUNC_MODEL="ct-punc-c"
NGPU="1"
DEVICE="cuda"
NCPU="4"
MAX_CONCURRENT="10"

# 启动服务
python server.py \
    --host $HOST \
    --port $PORT \
    --asr_model $ASR_MODEL \
    --vad_model $VAD_MODEL \
    --punc_model $PUNC_MODEL \
    --ngpu $NGPU \
    --device $DEVICE \
    --ncpu $NCPU \
    --max_concurrent_tasks $MAX_CONCURRENT

# 显示启动信息
echo ""
echo "FunASR 离线文本识别接口已启动！"
echo "服务地址: http://$HOST:$PORT"
echo "测试页面: http://$HOST:$PORT/"
echo ""
echo "可用接口:"
echo "- POST /submit_task    - 提交识别任务（文件上传或文件地址）"
echo "- GET  /get_task_status - 查询任务状态"
echo "- GET  /get_task_result - 查询任务结果"
echo "- POST /cancel_task    - 取消任务"
echo "- POST /delete_task    - 删除任务"
echo "- GET  /list_tasks     - 查询任务列表"
echo "- GET  /get_task_details - 查询任务详情"
echo "- POST /batch_operation - 批量操作"
echo "- POST /recognition    - 旧版兼容接口"
echo ""
echo "按 Ctrl+C 停止服务"
#!/bin/bash

# 创建日志文件夹
if [ ! -d "log/" ];then
  mkdir log
fi

# kill掉之前的进程
server_id=`ps -ef | grep server.py | grep -v "grep" | awk '{print $2}'`
echo $server_id

for id in $server_id
do
    kill -9 $id
    echo "killed $id"
done

# 启动多个服务，可以设置使用不同的显卡
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8001 >> log/output1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8002 >> log/output2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8003 >> log/output3.log 2>&1 &
