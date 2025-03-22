import sys

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="CosyVoice2 Demo")
parser.add_argument(
    "--model_dir",
    type=str,
    default="pretrained_models/CosyVoice2-0.5B",
    help="模型目录路径",
)
parser.add_argument(
    "--fp16", action="store_true", default=False, help="是否使用半精度(fp16)推理"
)
args = parser.parse_args()

print(f"使用模型目录: {args.model_dir}")
cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=args.fp16)

prompt_speech_16k = load_wav("./asset/sqr3.wav", 16000)
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        "。",
        "我会把三段话切成3段，用来做",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save(
        "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(
    cosyvoice.inference_cross_lingual(
        "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save(
        "fine_grained_control_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
    )

# instruct usage
for i, j in enumerate(
    cosyvoice.inference_instruct2(
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        "用四川话说这句话",
        prompt_speech_16k,
        stream=False,
    )
):
    torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
