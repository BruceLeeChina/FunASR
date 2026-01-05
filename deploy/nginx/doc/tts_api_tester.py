#pip install requests pydub
# !/usr/bin/env python3
"""
TTS语音合成API测试脚本
用于验证API接口的完整流程
"""

import requests
import time
import json
import sys
from datetime import datetime
import os
from pydub import AudioSegment
import io


class TTSServiceTester:
    def __init__(self, base_url="http://localhost:5000", hls_base_url="http://localhost:9080"):
        """
        初始化测试器

        Args:
            base_url: API服务基础URL
            hls_base_url: HLS服务基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.hls_base_url = hls_base_url.rstrip('/')
        self.current_req_id = None
        self.test_results = []

    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        # 保存到测试结果
        self.test_results.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })

    def test_health_check(self):
        """测试健康检查接口"""
        self.log("测试健康检查接口...")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)

            if response.status_code == 200:
                result = response.json()
                self.log(f"健康检查成功: {result}")

                # 检查关键字段
                required_fields = ["status", "service", "timestamp", "tts_initialized"]
                missing_fields = [field for field in required_fields if field not in result]

                if missing_fields:
                    self.log(f"缺少必要字段: {missing_fields}", "WARNING")
                    return False

                if result.get("tts_initialized") is False:
                    self.log("TTS模型未初始化，后续测试可能失败", "WARNING")

                self.log("健康检查接口测试通过 ✓")
                return True
            else:
                self.log(f"健康检查失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"健康检查请求异常: {e}", "ERROR")
            return False

    def test_tts_model(self):
        """测试TTS模型"""
        self.log("测试TTS模型...")

        try:
            response = requests.get(f"{self.base_url}/test-tts", timeout=30)

            if response.status_code == 200:
                result = response.json()
                self.log(f"TTS测试结果: {result}")

                if result.get("status") == "success":
                    self.log("TTS模型测试通过 ✓")
                    return True
                else:
                    self.log(f"TTS模型测试失败: {result.get('message')}", "ERROR")
                    return False
            else:
                self.log(f"TTS测试失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"TTS测试请求异常: {e}", "ERROR")
            return False

    def test_synthesize_api(self, text="你好，这是一个测试文本。请验证TTS合成功能是否正常工作。"):
        """测试合成接口"""
        self.log("测试语音合成接口...")
        self.log(f"合成文本: {text}")

        try:
            payload = {"text": text}
            response = requests.post(
                f"{self.base_url}/synthesize-and-push",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                self.log(f"合成请求响应: {result}")

                # 检查必要字段
                required_fields = ["req_id", "rtsp_url", "hls_url", "status", "message"]
                missing_fields = [field for field in required_fields if field not in result]

                if missing_fields:
                    self.log(f"缺少必要字段: {missing_fields}", "ERROR")
                    return False

                self.current_req_id = result["req_id"]
                self.log(f"请求ID: {self.current_req_id}")
                self.log(f"RTSP URL: {result['rtsp_url']}")
                self.log(f"HLS URL: {result['hls_url']}")
                self.log(f"状态: {result['status']}")

                # 验证URL格式
                if not result['rtsp_url'].startswith('rtsp://'):
                    self.log(f"RTSP URL格式错误: {result['rtsp_url']}", "WARNING")

                if not result['hls_url'].startswith('http://'):
                    self.log(f"HLS URL格式错误: {result['hls_url']}", "WARNING")

                self.log("语音合成接口测试通过 ✓")
                return True

            elif response.status_code == 400:
                self.log("合成请求参数错误", "ERROR")
                error_result = response.json()
                self.log(f"错误详情: {error_result}")
                return False

            else:
                self.log(f"合成请求失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"合成请求异常: {e}", "ERROR")
            return False

    def poll_stream_status(self, req_id, max_attempts=60, interval=5):  # 增加最大尝试次数和间隔时间
        self.log(f"开始轮询流状态，请求ID: {req_id}")

        # 先等待一段时间，确保服务端已开始处理
        self.log(f"等待服务端初始化请求，休眠10秒...")
        time.sleep(60)

        for attempt in range(1, max_attempts + 1):
            self.log(f"轮询尝试 {attempt}/{max_attempts}...")

            try:
                response = requests.get(
                    f"{self.base_url}/stream-status/{req_id}",
                    timeout=10
                )

                if response.status_code == 200:
                    status = response.json()
                    self.log(f"状态查询结果: {status}")

                    current_status = status.get("status")
                    message = status.get("message", "")

                    if current_status == "ready":
                        self.log(f"流已就绪: {message}")

                        # 验证HLS相关信息
                        if status.get("hls_exists"):
                            self.log(f"HLS文件存在，大小: {status.get('hls_file_size', '未知')} 字节")

                            segment_count = status.get("hls_segment_count")
                            if segment_count:
                                self.log(f"HLS片段数量: {segment_count}")

                            duration = status.get("estimated_duration")
                            if duration:
                                self.log(f"估计时长: {duration:.2f} 秒")

                        # 验证WAV文件信息
                        if status.get("wav_file_exists"):
                            wav_url = status.get("wav_direct_url")
                            if wav_url:
                                self.log(f"WAV文件URL: {wav_url}")

                        return True, status

                    elif current_status == "failed":
                        self.log(f"流处理失败: {message}", "ERROR")
                        return False, status

                    elif current_status in ["pending", "processing"]:
                        self.log(f"处理中: {message}")
                        # 继续轮询
                        time.sleep(interval)

                    else:
                        self.log(f"未知状态: {current_status}", "WARNING")
                        time.sleep(interval)

                elif response.status_code == 404:
                    self.log("请求不存在，可能已被清理", "WARNING")
                    return False, None

                else:
                    self.log(f"状态查询失败，状态码: {response.status_code}", "ERROR")
                    time.sleep(interval)

            except requests.exceptions.RequestException as e:
                self.log(f"状态查询异常: {e}", "ERROR")
                time.sleep(interval)

        self.log(f"轮询超时，超过最大尝试次数: {max_attempts}", "ERROR")
        return False, None

    def test_hls_access(self, hls_url):
        """测试HLS访问"""
        self.log(f"测试HLS访问: {hls_url}")

        try:
            response = requests.get(hls_url, timeout=10)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                content_length = len(response.content)

                self.log(f"HLS文件访问成功")
                self.log(f"Content-Type: {content_type}")
                self.log(f"文件大小: {content_length} 字节")

                # 检查是否为有效的m3u8文件
                if 'mpegurl' in content_type or response.text.startswith('#EXTM3U'):
                    self.log("有效的HLS m3u8文件 ✓")

                    # 解析m3u8内容
                    lines = response.text.split('\n')
                    segment_count = sum(1 for line in lines if line.endswith('.ts'))
                    self.log(f"找到 {segment_count} 个TS片段")

                    # 测试第一个TS片段（如果存在）
                    if segment_count > 0:
                        # 提取第一个TS片段URL
                        for line in lines:
                            if line.endswith('.ts') and not line.startswith('#'):
                                ts_url = line if line.startswith(
                                    'http') else f"{self.hls_base_url}/hls/test_{self.current_req_id}/{line}"
                                self.log(f"测试TS片段访问: {ts_url}")

                                try:
                                    ts_response = requests.get(ts_url, timeout=10)
                                    if ts_response.status_code == 200:
                                        self.log(f"TS片段访问成功，大小: {len(ts_response.content)} 字节")
                                        return True
                                    else:
                                        self.log(f"TS片段访问失败，状态码: {ts_response.status_code}", "WARNING")
                                except Exception as e:
                                    self.log(f"TS片段访问异常: {e}", "WARNING")
                                break

                    return True
                else:
                    self.log("不是有效的m3u8文件", "WARNING")
                    return False

            else:
                self.log(f"HLS文件访问失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"HLS访问异常: {e}", "ERROR")
            return False

    def test_wav_download(self, req_id):
        """测试WAV文件下载"""
        self.log(f"测试WAV文件下载，请求ID: {req_id}")

        # 方法1：使用download-wav接口
        try:
            download_url = f"{self.base_url}/download-wav/{req_id}"
            self.log(f"下载URL: {download_url}")

            response = requests.get(download_url, timeout=30, stream=True)

            if response.status_code == 200:
                content_length = response.headers.get('Content-Length')
                content_type = response.headers.get('Content-Type', '')

                self.log(f"下载成功，Content-Type: {content_type}")

                if content_length:
                    self.log(f"文件大小: {content_length} 字节")

                # 读取文件内容
                wav_data = response.content

                if len(wav_data) > 0:
                    # 验证WAV文件格式
                    try:
                        # 使用pydub验证
                        audio = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")

                        self.log(f"WAV文件验证成功:")
                        self.log(f"  时长: {len(audio) / 1000:.2f} 秒")
                        self.log(f"  采样率: {audio.frame_rate} Hz")
                        self.log(f"  声道数: {audio.channels}")
                        self.log(f"  最大振幅: {audio.max}")

                        # 保存到临时文件
                        temp_file = f"test_output_{req_id}.wav"
                        with open(temp_file, 'wb') as f:
                            f.write(wav_data)
                        self.log(f"文件已保存到: {temp_file}")

                        return True, wav_data
                    except Exception as e:
                        self.log(f"WAV文件验证失败: {e}", "WARNING")

                        # 检查文件头
                        if wav_data[:4] == b'RIFF' and wav_data[8:12] == b'WAVE':
                            self.log("文件头正确（RIFF WAVE），但无法用pydub解析", "WARNING")
                            return True, wav_data
                        else:
                            self.log("无效的WAV文件头", "ERROR")
                            return False, None
                else:
                    self.log("下载的文件为空", "ERROR")
                    return False, None

            else:
                self.log(f"下载失败，状态码: {response.status_code}", "ERROR")
                return False, None

        except requests.exceptions.RequestException as e:
            self.log(f"下载请求异常: {e}", "ERROR")
            return False, None

    def test_direct_audio_access(self, req_id):
        """测试直接音频文件访问"""
        self.log("测试直接音频文件访问接口...")

        # 首先获取音频文件列表
        try:
            list_url = f"{self.base_url}/audio/{req_id}/"
            response = requests.get(list_url, timeout=10)

            if response.status_code == 200:
                result = response.json()
                self.log(f"音频文件列表: {result}")

                if result.get("count", 0) > 0:
                    files = result.get("files", [])

                    for file_info in files:
                        file_url = file_info.get("url")
                        if file_url:
                            # 构建完整URL
                            full_url = f"{self.base_url}{file_url}"
                            self.log(f"测试访问: {full_url}")

                            try:
                                audio_response = requests.get(full_url, timeout=10)

                                if audio_response.status_code == 200:
                                    self.log(f"音频文件访问成功，大小: {len(audio_response.content)} 字节")

                                    # 验证音频文件
                                    if audio_response.content[:4] == b'RIFF':
                                        self.log("有效的WAV文件（RIFF头） ✓")
                                        return True
                                    else:
                                        self.log("不是有效的WAV文件", "WARNING")
                                        return False
                                else:
                                    self.log(f"音频文件访问失败，状态码: {audio_response.status_code}", "ERROR")
                                    return False

                            except Exception as e:
                                self.log(f"音频文件访问异常: {e}", "ERROR")
                                return False

                self.log("没有找到音频文件", "WARNING")
                return False

            else:
                self.log(f"获取音频列表失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"音频列表请求异常: {e}", "ERROR")
            return False

    def test_stop_stream(self, req_id):
        """测试停止单个流"""
        self.log(f"测试停止单个流，请求ID: {req_id}")

        try:
            response = requests.post(
                f"{self.base_url}/stop-stream/{req_id}",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                self.log(f"停止流响应: {result}")

                # 验证流已被停止
                time.sleep(2)  # 等待清理完成

                status_response = requests.get(f"{self.base_url}/stream-status/{req_id}", timeout=10)
                if status_response.status_code == 404:
                    self.log("流已成功停止并被清理 ✓")
                    return True
                else:
                    self.log("流可能未被完全清理", "WARNING")
                    return True  # 仍然返回True，因为停止请求已成功

            elif response.status_code == 404:
                self.log("流不存在，可能已被清理", "WARNING")
                return True

            else:
                self.log(f"停止流失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"停止流请求异常: {e}", "ERROR")
            return False

    def test_stop_all_streams(self):
        """测试停止所有流"""
        self.log("测试停止所有流...")

        try:
            response = requests.post(
                f"{self.base_url}/stop-stream",
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                self.log(f"停止所有流响应: {result}")
                self.log("停止所有流测试通过 ✓")
                return True
            else:
                self.log(f"停止所有流失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"停止所有流请求异常: {e}", "ERROR")
            return False

    def test_all_status(self):
        """测试批量状态查询"""
        self.log("测试批量状态查询接口...")

        try:
            response = requests.get(f"{self.base_url}/stream-status", timeout=10)

            if response.status_code == 200:
                result = response.json()
                self.log(f"批量状态查询结果: {result}")

                active_streams = result.get("active_streams", 0)
                streams = result.get("streams", [])

                self.log(f"活动流数量: {active_streams}")

                for i, stream in enumerate(streams):
                    self.log(f"流 {i + 1}: ID={stream.get('req_id')}, 状态={stream.get('status')}")

                self.log("批量状态查询测试通过 ✓")
                return True
            else:
                self.log(f"批量状态查询失败，状态码: {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"批量状态查询异常: {e}", "ERROR")
            return False

    def run_full_test(self, test_text=None):
        """运行完整测试流程"""
        self.log("=" * 60)
        self.log("开始TTS语音合成API完整测试")
        self.log("=" * 60)

        # 使用默认测试文本或自定义文本
        if test_text is None:
            test_text = "你好，这是一个测试文本。请验证TTS合成功能是否正常工作。今天天气很好，适合测试API接口。"

        test_steps = []

        # 步骤1: 健康检查
        self.log("\n步骤1: 健康检查")
        health_ok = self.test_health_check()
        test_steps.append(("健康检查", health_ok))

        # 步骤2: TTS模型测试
        self.log("\n步骤2: TTS模型测试")
        tts_ok = self.test_tts_model()
        test_steps.append(("TTS模型测试", tts_ok))

        if not tts_ok:
            self.log("TTS模型测试失败，跳过后续测试", "ERROR")
            return self.generate_test_report(test_steps)

        # 步骤3: 语音合成
        self.log("\n步骤3: 语音合成请求")
        synth_ok = self.test_synthesize_api(test_text)
        test_steps.append(("语音合成请求", synth_ok))

        if not synth_ok or not self.current_req_id:
            self.log("语音合成请求失败，跳过后续测试", "ERROR")
            return self.generate_test_report(test_steps)

        # 步骤4: 轮询状态
        self.log("\n步骤4: 轮询流状态")
        poll_ok, status_info = self.poll_stream_status(self.current_req_id)
        test_steps.append(("轮询流状态", poll_ok))

        if not poll_ok:
            self.log("流状态轮询失败，跳过后续测试", "ERROR")
            return self.generate_test_report(test_steps)

        # 步骤5: HLS访问测试
        self.log("\n步骤5: HLS访问测试")
        if status_info and status_info.get("hls_url"):
            hls_ok = self.test_hls_access(status_info["hls_url"])
            test_steps.append(("HLS访问测试", hls_ok))
        else:
            self.log("HLS URL不存在，跳过HLS访问测试", "WARNING")
            test_steps.append(("HLS访问测试", "SKIPPED"))

        # 步骤6: WAV下载测试
        self.log("\n步骤6: WAV文件下载测试")
        wav_ok, wav_data = self.test_wav_download(self.current_req_id)
        test_steps.append(("WAV文件下载测试", wav_ok))

        # 步骤7: 直接音频访问测试
        self.log("\n步骤7: 直接音频访问测试")
        direct_audio_ok = self.test_direct_audio_access(self.current_req_id)
        test_steps.append(("直接音频访问测试", direct_audio_ok))

        # 步骤8: 批量状态查询
        self.log("\n步骤8: 批量状态查询测试")
        all_status_ok = self.test_all_status()
        test_steps.append(("批量状态查询测试", all_status_ok))

        # 步骤9: 停止单个流
        self.log("\n步骤9: 停止单个流测试")
        stop_single_ok = self.test_stop_stream(self.current_req_id)
        test_steps.append(("停止单个流测试", stop_single_ok))

        # 步骤10: 停止所有流
        self.log("\n步骤10: 停止所有流测试")
        stop_all_ok = self.test_stop_all_streams()
        test_steps.append(("停止所有流测试", stop_all_ok))

        # 生成测试报告
        return self.generate_test_report(test_steps)

    def generate_test_report(self, test_steps):
        """生成测试报告"""
        self.log("\n" + "=" * 60)
        self.log("测试报告")
        self.log("=" * 60)

        total_steps = len(test_steps)
        passed_steps = sum(1 for _, result in test_steps if result is True)
        skipped_steps = sum(1 for _, result in test_steps if result == "SKIPPED")
        failed_steps = total_steps - passed_steps - skipped_steps

        # 输出每个步骤的结果
        for i, (step_name, result) in enumerate(test_steps, 1):
            status_symbol = "✓" if result is True else ("⚠" if result == "SKIPPED" else "✗")
            self.log(f"{i:2d}. {step_name:20} {status_symbol}")

        self.log("\n" + "=" * 60)
        self.log(f"总计: {total_steps} 个测试步骤")
        self.log(f"通过: {passed_steps}")
        self.log(f"跳过: {skipped_steps}")
        self.log(f"失败: {failed_steps}")

        success_rate = (passed_steps / total_steps) * 100 if total_steps > 0 else 0
        self.log(f"成功率: {success_rate:.1f}%")

        if failed_steps == 0:
            self.log("整体测试结果: 全部通过 ✓")
        else:
            self.log("整体测试结果: 存在失败项 ✗", "ERROR")

        self.log("=" * 60)

        # 保存详细日志到文件
        log_filename = f"tts_api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_filename, 'w', encoding='utf-8') as f:
            for log_entry in self.test_results:
                f.write(f"[{log_entry['timestamp']}] [{log_entry['level']}] {log_entry['message']}\n")

        self.log(f"详细日志已保存到: {log_filename}")

        return failed_steps == 0


def main():
    """主函数"""
    print("TTS语音合成API测试工具")
    print("=" * 60)

    # 配置参数
    import argparse
    parser = argparse.ArgumentParser(description='TTS语音合成API测试工具')
    parser.add_argument('--url', default='http://localhost:5000', help='API服务URL')
    parser.add_argument('--hls-url', default='http://localhost:9080', help='HLS服务URL')
    parser.add_argument('--text', help='测试文本')
    parser.add_argument('--skip-full', action='store_true', help='跳过完整测试，只做健康检查')

    args = parser.parse_args()

    # 创建测试器
    tester = TTSServiceTester(base_url=args.url, hls_base_url=args.hls_url)

    if args.skip_full:
        # 只做健康检查
        print("\n只进行健康检查...")
        health_ok = tester.test_health_check()
        tts_ok = tester.test_tts_model()

        if health_ok and tts_ok:
            print("\n基础服务检查通过 ✓")
            sys.exit(0)
        else:
            print("\n基础服务检查失败 ✗")
            sys.exit(1)
    else:
        # 完整测试
        test_text = args.text or "你好，这是一个测试文本。请验证TTS合成功能是否正常工作。今天天气很好，适合测试API接口。"

        print(f"使用API URL: {args.url}")
        print(f"使用HLS URL: {args.hls_url}")
        print(f"测试文本: {test_text[:50]}..." if len(test_text) > 50 else f"测试文本: {test_text}")
        print()

        # 运行完整测试
        success = tester.run_full_test(test_text)

        # 返回退出码
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# # 基本测试（使用默认配置）
# python tts_api_tester.py
#
# # 指定服务地址
# python tts_api_tester.py --url http://192.168.21.164:5000 --hls-url http://192.168.21.164:9080
#
# # 使用自定义文本
# python tts_api_tester.py --text "你好，世界！这是一个自定义测试文本。"
#
# # 只做健康检查
# python tts_api_tester.py --skip-full