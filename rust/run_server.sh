#!/bin/bash
# Run both Python backend and Rust server with GPU auto-optimization
# Usage: ./run_server.sh [--model-dir <path>] [--auto-optimize] [--fp16] [--trt]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-pretrained_models/Fun-CosyVoice3-0.5B}"
PYTHON_PORT=50000
GRPC_PORT=50051

# Default optimization flags (auto-detected)
AUTO_OPTIMIZE=true
FP16=""
LOAD_TRT=""
TRT_CONCURRENT=""
LOAD_VLLM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --no-auto-optimize)
            AUTO_OPTIMIZE=false
            shift
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --trt)
            LOAD_TRT="--load-trt"
            shift
            ;;
        --trt-concurrent)
            TRT_CONCURRENT="--trt-concurrent=$2"
            shift 2
            ;;
        --vllm)
            LOAD_VLLM="--load-vllm"
            shift
            ;;
        --python-port)
            PYTHON_PORT="$2"
            shift 2
            ;;
        --grpc-port)
            GRPC_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              CosyVoice Server Stack                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Python backend: http://localhost:$PYTHON_PORT"
echo "  gRPC server:    http://localhost:$GRPC_PORT"
echo "  Model:          $MODEL_DIR"
echo ""

# GPU Detection and Auto-Optimization
if $AUTO_OPTIMIZE; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              GPU Auto-Optimization                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "")

        if [ -n "$GPU_INFO" ]; then
            GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            GPU_VRAM=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
            GPU_CC=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)

            echo "  GPU Detected: $GPU_NAME"
            echo "  VRAM: ${GPU_VRAM} MiB"
            echo "  Compute Capability: $GPU_CC"
            echo ""

            # Auto-configure based on GPU
            # RTX 20xx, 30xx, 40xx all support FP16 + TensorRT
            if [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"V100"* ]]; then
                echo "  → Enabling FP16 (Tensor cores detected)"
                FP16="--fp16"

                echo "  → Enabling TensorRT acceleration"
                LOAD_TRT="--load-trt"

                # Set TRT concurrent based on VRAM
                if [ "$GPU_VRAM" -lt 6000 ]; then
                    TRT_CONCURRENT="--trt-concurrent=1"
                    echo "  → TRT concurrent: 1 (low VRAM)"
                elif [ "$GPU_VRAM" -lt 10000 ]; then
                    TRT_CONCURRENT="--trt-concurrent=2"
                    echo "  → TRT concurrent: 2 (medium VRAM)"
                elif [ "$GPU_VRAM" -lt 16000 ]; then
                    TRT_CONCURRENT="--trt-concurrent=3"
                    echo "  → TRT concurrent: 3 (good VRAM)"
                else
                    TRT_CONCURRENT="--trt-concurrent=4"
                    echo "  → TRT concurrent: 4 (high VRAM)"
                fi

                # Enable vLLM for high-end GPUs with lots of VRAM
                if [ "$GPU_VRAM" -ge 12000 ]; then
                    # Only for RTX 30xx and 40xx series
                    if [[ "$GPU_NAME" == *"RTX 30"* ]] || [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"A100"* ]]; then
                        echo "  → Enabling vLLM acceleration"
                        LOAD_VLLM="--load-vllm"
                    fi
                fi
            elif [[ "$GPU_NAME" == *"GTX 16"* ]]; then
                # GTX 16xx: Turing without tensor cores
                echo "  → GTX 16xx detected: TensorRT only (no tensor cores)"
                LOAD_TRT="--load-trt"
                TRT_CONCURRENT="--trt-concurrent=1"
            elif [[ "$GPU_NAME" == *"GTX 10"* ]]; then
                # GTX 10xx: Pascal
                echo "  → GTX 10xx detected: Basic TensorRT mode"
                LOAD_TRT="--load-trt"
                TRT_CONCURRENT="--trt-concurrent=1"
            else
                echo "  → Unknown GPU: Using default configuration"
            fi
        else
            echo "  ⚠ No NVIDIA GPU detected"
        fi
    else
        echo "  ⚠ nvidia-smi not found - GPU auto-optimization disabled"
    fi
    echo ""
fi

# Build optimization flags string
PYTHON_OPTS=""
[ -n "$FP16" ] && PYTHON_OPTS="$PYTHON_OPTS $FP16"
[ -n "$LOAD_TRT" ] && PYTHON_OPTS="$PYTHON_OPTS $LOAD_TRT"
[ -n "$TRT_CONCURRENT" ] && PYTHON_OPTS="$PYTHON_OPTS $TRT_CONCURRENT"
[ -n "$LOAD_VLLM" ] && PYTHON_OPTS="$PYTHON_OPTS $LOAD_VLLM"

echo "Python options: $PYTHON_OPTS"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $PYTHON_PID 2>/dev/null || true
    kill $RUST_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Python FastAPI backend
echo "Starting Python backend..."
pixi run python runtime/python/fastapi/server.py \
    --port $PYTHON_PORT \
    --model_dir "$MODEL_DIR" \
    $PYTHON_OPTS &
PYTHON_PID=$!

# Wait for Python backend to start
echo "Waiting for Python backend..."
MAX_WAIT=60
for i in $(seq 1 $MAX_WAIT); do
    if curl -s "http://localhost:$PYTHON_PORT/health" > /dev/null 2>&1; then
        echo "Python backend ready!"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "ERROR: Python backend failed to start within ${MAX_WAIT}s"
        kill $PYTHON_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Start Rust gRPC server
echo "Starting Rust gRPC server..."
"$SCRIPT_DIR/target/release/cosyvoice-server" \
    --port $GRPC_PORT \
    --python-backend "http://localhost:$PYTHON_PORT" &
RUST_PID=$!

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Server Stack Running                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Python API:  http://localhost:$PYTHON_PORT                     ║"
echo "║  gRPC Server: localhost:$GRPC_PORT                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Endpoints:                                                  ║"
echo "║    GET  /health          - Health check                      ║"
echo "║    GET  /gpu_stats       - GPU statistics                    ║"
echo "║    GET  /performance_stats - Performance metrics             ║"
echo "║    POST /benchmark       - Run RTF benchmark                 ║"
echo "║    GET  /speakers        - List available speakers           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop."
wait
