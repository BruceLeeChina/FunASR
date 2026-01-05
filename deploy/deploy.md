# D:\workspace\PycharmProjects\FunASR\deploy --> /opt
# D:\workspace\PycharmProjects\FunASR\runtime\python\http --> /opt/funasr-offline
# D:\workspace\PycharmProjects\tts-gpt-sovits  --> /opt/tts
# D:\largemodel\TTS-ASR\funasr --> /opt/funasr-online


cd /opt && docker-compose up -d
docker-compose up -d

语音识别+语音合成：
1. TTS（旧版本）：
http://192.168.21.164:5000/
文档：
http://192.168.21.164/doc/tts_api_doc.html

2. ASR在线：参考 README_zh.md
访问地址：https://192.168.21.164/
websocket
wss://192.168.21.164:10095/ws
文档：
http://192.168.21.164/doc/asr_api_doc.html

3. ASR离线：
http://192.168.21.164:8000/
文档：
http://192.168.21.164/doc/asr_offline_api_doc.html



============================================================================
===========语音识别 在线 ASR==============
测试demo火狐浏览器访问：  
静态页面访问： https://192.168.21.130:84/

websocket：wss://192.168.21.130:10095/
证书验证绕过：https://192.168.21.130:10095/

使用：点击左下角“连接”按钮，然后点击“开始”按钮，说话进行识别

===========语音识别 离线 ASR==============
ASR离线识别地址：
Nginx访问：
http://192.168.21.130:7080/

直接访问：
http://192.168.21.130:8000

接口文档：
http://192.168.21.130:7080/doc/

测试：单任务提交-->点击选择或拖放音频文件


===========语音合成 TTS==============
语音合成：TTS
http://192.168.21.130:9080/
使用： 输入文本 -->  点击 "合成并开始新流" 按钮 --> 合成后自动播放 --> 播放页面 “⬇ 下载WAV” 可以下载语音文件

接口文档：
语音合成：TTS
http://192.168.21.130:9080/doc/tts_api_doc.html
ASR：
http://192.168.21.130:9080/doc/asr_api_doc.html