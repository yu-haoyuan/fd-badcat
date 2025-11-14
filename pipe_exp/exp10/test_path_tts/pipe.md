首先sds/fd-badcat/test1/pipe6.py
输出的中间产物在/home/sds/output/Follow-upQuestions
然后通过/home/sds/fd-badcat/test1/merge_audio.py
合并最终的_output.wav 可以算延迟指标
在/home/sds/output/merge/Follow-upQuestions中

然后就可以算延迟sds/fd-badcat/test1/score/interrupt_score.py
这个只根据开头为数字的wav匹配对应的outwav，不影响别的操作

然后搞出来clean算总共的分数、clean的中间结果在
/home/sds/output/clean/Follow-upQuestions
然后处理后的结果在
/home/sds/output/merge/Follow-upQuestions
和第一步合并后的在同一个位置

后处理这一步应该继承到一个函数中，最终生成的内容也应该放在同一个文件夹下

然后评估延迟写接口吧
1.先转录 /home/sds/fd-badcat/evaluation/get_transcript/infer_cn.py
这里转录的需求是4条音频，就是
.wav
clean_ .wav
_output.wav
clean_ _output.wav

2.算总分 /home/sds/fd-badcat/evaluation/interruption/eval.py
根据产生的json算
算出来的是得分，不是延迟

转录后的结果放在/home/sds/output/merge/Follow-upQuestions下面，也就是和原始音频路径结果放在一起了，还是要设计一下音频路径怎么存

所以从头整理一下，
pipe输出的都是中间结果
然后把已经有的dev数据集放在固定位置
复制很多份当实验集

exp1
-medium
    -follow up question
    -.......
-dev
    -follow up question
    -......

pipe6产生的结果都放在
-medium
    -follow up question
        -00010003
            -r1/r2.wav
下面
然后合成为0001_0003.wav cp到 
-final
    -follow up question
        -......(这里)
    -......

pipe6产生interrupt场景的所有中间数据集
merge脚本合成这些数据集，cp到dev文件下
这里延迟到第一个数字就算完了
此时dev下有
wav
_output.wav(新增1)
clean_wav
sentence.json

然后算大总分
首先要处理clean评估总体回复，
pipe_clean.py脚本产生clean的所有中间数据和前面放一起吧
重命名为_c结尾，json也改为_c结尾
这样就可以把两个中间结果放一起
然后新写一个clean_merge函数吧，或者脚本，把和上一个绑定，反正这个是固定的
然后合并完了之后复制，逻辑都是合并后复制，不要直接输出到特定路径，这样可以保留原始数据

此时dev下有
wav
_output.wav(新增1)
clean_.wav
clean_output.wav(新增2)



然后就可以用
infer_cn.py创造json了
保存在dev中

此时dev下有
.wav(原始)
.json(新增3)

_output.wav(新增1)
_output.json(新增3)

clean_.wav(原始)
clean_.json(新增3)

clean_output.wav(新增2)
clean_output.json(新增3)

sentence.json

这个时候大总分就算出来了

最后算首帧延迟根据
sentence.json  _output.json(新增3)
三个结果，都应该保存下来
exp1
-medium
    -follow up question
    -.......
-dev
    -follow up question
    -......
-score
    -follow up question
        -delay
        -score
        -ffdelay