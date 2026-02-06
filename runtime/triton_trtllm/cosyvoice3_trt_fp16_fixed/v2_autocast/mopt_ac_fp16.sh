onnx_path="/weights/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx"
calibration_data="/opt/code/CosyVoice/cosyvoice_trt_dev/mopt/autocast/cus_pg_inputs.json"
output_path="/weights/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.autocast_fp16.onnx"
low_precision_type=fp16
log_level=INFO
data_max=65504

python -m modelopt.onnx.autocast \
    --onnx_path ${onnx_path} \
    --output_path ${output_path} \
    --low_precision_type ${low_precision_type} \
    --calibration_data ${calibration_data} \
    --log_level ${log_level} \
    --data_max ${data_max} \
