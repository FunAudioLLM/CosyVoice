#!/bin/bash
# Robust server restart - uses setsid to fully detach from parent session
pkill -9 -f server_cosyvoice3 2>/dev/null || true
pkill -9 -f run_server 2>/dev/null || true
sleep 3
> /home/zhiqiang/server-opt.log
setsid bash -c '/home/zhiqiang/run_server.sh > /home/zhiqiang/server-opt.log 2>&1' < /dev/null > /dev/null 2>&1 &
echo "launched pid=$!"
