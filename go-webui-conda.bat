@echo off
REM 定义 Anaconda 路径
set CONDA_PATH=D:\ProgramData\anaconda3

REM 激活目标虚拟环境
CALL "%CONDA_PATH%\condabin\conda.bat" activate "D:\AI\CosyVoice-main\venv"

REM 检查是否激活成功
IF ERRORLEVEL 1 (
    echo 激活虚拟环境失败，请检查路径或环境名称！
    pause
    exit /b
)

REM 执行 Python 脚本
python api.py --webui True --port 9997 --model_dir pretrained_models/CosyVoice-300M-Instruct

REM 保持窗口打开
pause
