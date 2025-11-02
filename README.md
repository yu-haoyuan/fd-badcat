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

1101 测试
创建一个 exp/exp1/dev/dev_zh/Follow-up Questions
把Follow-up Questions复制粘贴到dev_zh即可

cd fd-badcat
python test_path_rej/inter_pipe.py

---
由于Full-Duplex-Bench v1.5不支持Pause_Handling场景，我们使用compute_rejection_rate.py计算拒识场景Pause_Handling的rejection_rate（保留三位小数，比如0.425，即42.5%）。在使用compute_rejection_rate.py前，请确保使用evaluation/get_transcript下的相应infer文件生成了所需的所有json文件。
compute_first_response_delay.py 计算拒识场景Pause_Handling的模型首次回复的延迟。

这个是
evaluation/rejection/Pause_Handling/compute_first_response_delay.py
和evaluation/rejection/Pause_Handling/compute_rejection_rate.py
的计算方法,应该直接调用就行了
看起来这里可以调用函数
from evaluation.rejection.Pause_Handling.compute_rejection_rate.process_folder
这个函数没有写入操作,需要自己实现

同样,这个pausl hand的ftd也不会写入操作

---
可以先整理所有的ftd操作,因为都是同一个函数
输入文件夹,输出json文件,放在同一个路径下
算了 不要改函数了,一律都是调用脚本
然后修改脚本,大方修改,写好脚本的输入和输出路径即可,由
test_path_rej/inter_score.py得到全部的分数
然后test_path_rej/inter_sum.py负责计算所有分数的最终结果,写入一个json中
可以后期合并这两个函数