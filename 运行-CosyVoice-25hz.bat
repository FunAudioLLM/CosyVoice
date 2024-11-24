SET PYTHON_PATH=%cd%\py311\
rem overriding default python env vars in order not to interfere with any system python installation
SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONWEXECUTABLE=%PYTHON_PATH%pythonw.exe
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONW_EXECUTABLE=%PYTHON_PATH%pythonw.exe
SET PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET FFMPEG_PATH=%cd%\py311\ffmpeg\bin
SET PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%FFMPEG_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download
set PYTHONPATH=third_party/AcademiCodec;third_party/Matcha-TTS
"%PYTHON_EXECUTABLE%" webui.py --port 9886 --model_dir pretrained_models/CosyVoice-300M-25Hz
pause
