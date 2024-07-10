#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

# 终止已经运行的相同脚本实例
if pgrep -f "auto_task.py" > /dev/null
then
    pgrep -f "auto_task.py" | xargs kill -9
    echo "Terminated existing instances of auto_task.py"
else
    echo "No existing instances of auto_task.py found"
fi

# 在后台运行新的脚本实例，并将输出重定向到 out.log
nohup /root/miniconda3/envs/cosyvoice/bin/python auto_task.py --book_name jy --start_idx 1 --end_idx 10 > out.log 2>&1 &
