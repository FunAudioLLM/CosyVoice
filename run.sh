#!/bin/bash
. ./path.sh || exit 1;

if pgrep -f "auto_task.py" > /dev/null
then
    pgrep -f "auto_task.py" | xargs kill -9
    echo "Terminated existing instances of auto_task.py"
else
    echo "No existing instances of auto_task.py found"
fi

# 在后台运行新的脚本实例，并将输出重定向到 out.log
nohup /root/miniconda3/envs/cosyvoice/bin/python auto_task.py --book_name fz --start_idx 201 --end_idx 375 > out.log 2>&1 &
# nohup /root/miniconda3/envs/cosyvoice/bin/python auto_task.py --book_name srzy --start_idx 174 --end_idx 200 > out.log 2>&1 &

# nohup /root/miniconda3/envs/cosyvoice/bin/python auto_task.py --book_name aa --start_idx 1 --end_idx 1 > out.log 2>&1 &
