#!/bin/bash
# Stable restart - NO LD_LIBRARY_PATH cudnn override (which destabilizes WSL)
pkill -9 -f server_cosyvoice3 2>/dev/null || true
sleep 3
> /home/zhiqiang/server-opt.log
cd /home/zhiqiang/repos/CosyVoice
setsid bash -c '/home/zhiqiang/.venvs/cosyvoice/bin/python -u server_cosyvoice3.py > /home/zhiqiang/server-opt.log 2>&1' < /dev/null > /dev/null 2>&1 &
echo "launched pid=$!"
