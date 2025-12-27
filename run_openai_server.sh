#!/bin/bash
MODEL_DIR="pretrained_models/Fun-CosyVoice3-0.5B"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model directory $MODEL_DIR not found. Downloading models..."
    source venv/bin/activate
    python download_models.py
else
    echo "Model directory found."
    source venv/bin/activate
fi

echo "Starting OpenAI-compatible server..."
# Check if uvicorn is installed, if not try pip install
python -c "import uvicorn" 2>/dev/null || pip install uvicorn fastapi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib
python openai_server.py --port 50000 --model_dir "$MODEL_DIR"
