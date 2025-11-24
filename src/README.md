由于是前后端运行，所以需要运行
```
sudo apt install tmux
bash src/sc.sh
```
如果运行失败，检查18000端口是否被占用

等待开始运行后自动进入前端界面

由于是现实时间模拟，运行时间和输入音频总时长相同

完毕后自动跳转到后端窗口显示

`INFO:connection closed`

手动`ctrl c`退出后端

然后运行
```
for d in exp/exp-1/HD-Track2/*; do echo "$(basename "$d"): $(find "$d" -maxdepth 1 -type f -name "*.wav" | wc -l)"; done
```
查看是否和输入文件数相同