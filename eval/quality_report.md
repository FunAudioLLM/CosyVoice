# Audio quality across optimization rounds

| round | n | avg CER | avg SECS | avg RMS dB | avg dur s | flag |
|---|---:|---:|---:|---:|---:|---|
| round0_baseline | 4 | 0.2536 | 0.6065 | -21.645 | 4.33 | - |
| round10_hift_fp32 | 4 | 0.2339 | 0.6153 | -20.29 | 4.11 | - |
| round1_fp16 | 5 | 0.1835 | 0.6719 | -20.026 | 7.744 | - |
| round2_vllm | 4 | 0.2143 | 0.6764 | -21.1125 | 4.12 | - |
| round3_lockfree | 4 | 0.2697 | 0.6617 | -20.415 | 4.19 | - |
| round6_hift_trt | 4 | 1.0 | -0.1377 | 0.0 | 4.53 | INTELLIGIBILITY  VOICE |
| round7_fixed | 4 | 0.2339 | 0.6151 | -20.29 | 4.11 | - |
| round7_flow_concurrent | 4 | 1.0 | -0.1376 | 0.0 | 4.27 | INTELLIGIBILITY  VOICE |
