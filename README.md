# fd-badcat
fd-sds

---
dabu 1013
cascade 方案  
llm：qwen2.5-0.6b-instruct，能说最简单的人话就行  
tts：中英文分开，延迟应该没有体感差别  
vad：silero vad  
asr：paraformer？暂定  
然后写一个websocket通过网页测试demo/批量上传测试数据，先跑通baseline，再考虑训练  

dabu1031
test_path是应对interrupt的,然后对接了evaluate的脚本,规范路径

test_rej继续拓展脚本,希望能做到处理停顿问题和bc问题

1101目前可以做到处理停顿问题和bc问题,还需要实现一个缓存记录历史对话,目前只有两轮暂不记录