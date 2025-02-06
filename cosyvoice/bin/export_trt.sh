#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
# download tensorrt from https://developer.nvidia.com/tensorrt/download/10x, check your system and cuda for compatibability
# for example for linux + cuda12.4, you can download https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
TRT_DIR=<YOUR_TRT_DIR>
MODEL_DIR=<YOUR_MODEL_DIR>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_DIR/lib:/usr/local/cuda/lib64

# cosyvoice export
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp32.mygpu.plan --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4 --optShapes=x:2x80x200,mask:2x1x200,mu:2x80x200,cond:2x80x200 --maxShapes=x:2x80x3000,mask:2x1x3000,mu:2x80x3000,cond:2x80x3000 --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw --outputIOFormats=fp32:chw
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan --fp16 --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4 --optShapes=x:2x80x200,mask:2x1x200,mu:2x80x200,cond:2x80x200 --maxShapes=x:2x80x3000,mask:2x1x3000,mu:2x80x3000,cond:2x80x3000 --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw --outputIOFormats=fp16:chw

# cosyvoice2 export with cache
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp32.mygpu.plan \
    --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4,down_blocks_kv_cache:1x4x2x0x512x2,mid_blocks_kv_cache:12x4x2x0x512x2,up_blocks_kv_cache:1x4x2x0x512x2 \
    --optShapes=x:2x80x200,mask:2x1x200,mu:2x80x200,cond:2x80x200,down_blocks_kv_cache:1x4x2x100x512x2,mid_blocks_kv_cache:12x4x2x100x512x2,up_blocks_kv_cache:1x4x2x100x512x2 \
    --maxShapes=x:2x80x1500,mask:2x1x1500,mu:2x80x1500,cond:2x80x1500,down_blocks_kv_cache:1x4x2x200x512x2,mid_blocks_kv_cache:12x4x2x200x512x2,up_blocks_kv_cache:1x4x2x200x512x2 \
    --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw \
    --outputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw
$TRT_DIR/bin/trtexec --onnx=$MODEL_DIR/flow.decoder.estimator.fp32.onnx --saveEngine=$MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan --fp16 \
    --minShapes=x:2x80x4,mask:2x1x4,mu:2x80x4,cond:2x80x4,down_blocks_kv_cache:1x4x2x0x512x2,mid_blocks_kv_cache:12x4x2x0x512x2,up_blocks_kv_cache:1x4x2x0x512x2 \
    --optShapes=x:2x80x200,mask:2x1x200,mu:2x80x200,cond:2x80x200,down_blocks_kv_cache:1x4x2x100x512x2,mid_blocks_kv_cache:12x4x2x100x512x2,up_blocks_kv_cache:1x4x2x100x512x2 \
    --maxShapes=x:2x80x1500,mask:2x1x1500,mu:2x80x1500,cond:2x80x1500,down_blocks_kv_cache:1x4x2x200x512x2,mid_blocks_kv_cache:12x4x2x200x512x2,up_blocks_kv_cache:1x4x2x200x512x2 \
    --inputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw \
    --outputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw
