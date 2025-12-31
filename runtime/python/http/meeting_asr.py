from funasr import AutoModel
import json
import os


def process_meeting_audio(audio_path):
    """
    处理会议音频并返回对话格式结果
    """
    print(f"正在处理音频: {audio_path}")
    
    # 初始化模型 - 支持说话人识别
    print("正在初始化模型...")
    model = AutoModel(
        model="paraformer-zh",  # ASR模型
        vad_model="fsmn-vad",  # 语音活动检测模型
        punc_model="ct-punc",  # 标点符号模型
        spk_model="cam++"  # 说话人识别模型
    )

    print("开始识别音频...")
    # 生成识别结果
    result = model.generate(
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


def format_meeting_dialogue(dialogue_list):
    """
    将识别结果格式化为清晰的会议对话形式，并合并连续说话者的内容
    """
    if not dialogue_list:
        return "未识别到任何对话内容"
    
    # 合并连续说话者的内容
    merged_dialogue = []
    if dialogue_list:
        current_speaker = dialogue_list[0]['speaker']
        current_text = dialogue_list[0]['text']
        current_start = dialogue_list[0]['start_time']
        current_end = dialogue_list[0]['end_time']
        
        for i in range(1, len(dialogue_list)):
            entry = dialogue_list[i]
            
            # 如果是同一说话人，合并内容
            if entry['speaker'] == current_speaker:
                current_text += entry['text']
                current_end = entry['end_time']
            else:
                # 不同说话人，保存当前内容并开始新的合并
                merged_dialogue.append({
                    'speaker': current_speaker,
                    'text': current_text,
                    'start_time': current_start,
                    'end_time': current_end
                })
                current_speaker = entry['speaker']
                current_text = entry['text']
                current_start = entry['start_time']
                current_end = entry['end_time']
        
        # 添加最后一个合并的段落
        merged_dialogue.append({
            'speaker': current_speaker,
            'text': current_text,
            'start_time': current_start,
            'end_time': current_end
        })
    
    # 格式化输出
    result = "=== 会议对话识别结果 ===\n"
    for i, entry in enumerate(merged_dialogue):
        speaker = entry['speaker']
        text = entry['text']
        start_time = entry['start_time']
        end_time = entry['end_time']
        
        result += f"[{start_time:.2f}s - {end_time:.2f}s] {speaker}: {text}\n"
        # 在说话人切换时添加空行
        if i < len(merged_dialogue) - 1 and merged_dialogue[i]['speaker'] != merged_dialogue[i+1]['speaker']:
            result += "\n"
    
    return result


# 使用示例
audio_file = "./data/meeting_recording.wav"  # 使用非空音频文件

# 检查文件是否存在
if os.path.exists(audio_file):
    print(f"音频文件存在，大小: {os.path.getsize(audio_file)} 字节")
    dialogue = process_meeting_audio(audio_file)
else:
    print(f"音频文件不存在: {audio_file}")
    dialogue = []

# 格式化并输出结果
formatted_output = format_meeting_dialogue(dialogue)
print(formatted_output)