# Flow Estimator TRTLLM Conversion

## Setup 
Download model
```python
# modelscope SDK model download
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# for oversea users, huggingface SDK model download
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

setup docker environment
```sh
docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
```

run the container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "cosyvoice-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-cosyvoice:25.06
```

## model conversion

convert checkpoint
```sh
python3 convert_checkpoint.py --pytorch_ckpt /workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/flow.pt 
```

build 
```sh
trtllm-build \
  --checkpoint_dir tllm_checkpoint \
  --model_cls_file dit_trt.py \
  --model_cls_name CosyVoiceDiT \
  --output_dir ./tllm_engine \
  --max_batch_size 8 \
  --max_seq_len 2000 \
  --remove_input_padding disable --bert_context_fmha_fp32_acc enable
```

The default built trt engine **DOES NOT SUPPORT STREAMING INFERENCE** because the `bert_attention` plugin does not accept `attention_mask` as part of input.
One could disable the plugin with `--bert_attention_plugin disable` and add attention mask. However, generated speech quality is lower in some scenarios.

One can also run the full conversion + example inference in the jupyter notebook `conversion.ipynb` directly. 


## Contact 
Ming Yang Zhou, Envision.AI (ming@envision.ai)
 


