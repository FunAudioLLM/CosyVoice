#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# download tensorrt from https://developer.nvidia.com/tensorrt/download/10x, check your system and cuda for compatibability
# for example for linux + cuda12.4, you can download https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
TRT_DIR=<YOUR_TRT_DIR>
MODEL_DIR=<YOUR_MODEL_DIR>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_DIR/lib:/usr/local/cuda/lib64

# cosyvoice export
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp32.mygpu.plan --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4 --optShapes=x:2x80x193,mask:2x1x193,mu:2x80x193,cond:2x80x193 --maxShapes=x:2x80x6800,mask:2x1x6800,mu:2x80x6800,cond:2x80x6800 --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw --outputIOFormats=fp32:chw
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan --fp16 --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4 --optShapes=x:2x80x193,mask:2x1x193,mu:2x80x193,cond:2x80x193 --maxShapes=x:2x80x6800,mask:2x1x6800,mu:2x80x6800,cond:2x80x6800 --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw --outputIOFormats=fp16:chw

# cosyvoice2 export with cache
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp32.mygpu.plan \
    --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4,down_blocks_kv_cache:1x4x2x0x512x2,mid_blocks_kv_cache:12x4x2x0x512x2,up_blocks_kv_cache:1x4x2x0x512x2 \
    --optShapes=x:2x80x193,mask:2x1x193,mu:2x80x193,cond:2x80x193,down_blocks_kv_cache:1x4x2x193x512x2,mid_blocks_kv_cache:12x4x2x193x512x2,up_blocks_kv_cache:1x4x2x193x512x2 \
    --maxShapes=x:2x80x6800,mask:2x1x6800,mu:2x80x6800,cond:2x80x6800,down_blocks_kv_cache:1x4x2x200x512x2,mid_blocks_kv_cache:12x4x2x200x512x2,up_blocks_kv_cache:1x4x2x200x512x2 \
    --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw \
    --outputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan --fp16 \
    --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4,down_blocks_kv_cache:1x4x2x0x512x2,mid_blocks_kv_cache:12x4x2x0x512x2,up_blocks_kv_cache:1x4x2x0x512x2 \
    --optShapes=x:2x80x193,mask:2x1x193,mu:2x80x193,cond:2x80x193,down_blocks_kv_cache:1x4x2x193x512x2,mid_blocks_kv_cache:12x4x2x193x512x2,up_blocks_kv_cache:1x4x2x193x512x2 \
    --maxShapes=x:2x80x6800,mask:2x1x6800,mu:2x80x6800,cond:2x80x6800,down_blocks_kv_cache:1x4x2x200x512x2,mid_blocks_kv_cache:12x4x2x200x512x2,up_blocks_kv_cache:1x4x2x200x512x2 \
    --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw \
    --outputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw
