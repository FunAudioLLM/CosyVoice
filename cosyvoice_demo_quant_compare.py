import sys

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import argparse
import os

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="CosyVoice2 Demo")
parser.add_argument(
    "--model_dir",
    type=str,
    default="pretrained_models/CosyVoice2-0.5B",
    help="模型目录路径",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="exp",
    help="输出目录路径",
)

parser.add_argument(
    "--fp16", action="store_true", default=False, help="是否使用半精度(fp16)推理"
)
args = parser.parse_args()

print(f"使用模型目录: {args.model_dir}")
cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=args.fp16)
# cosyvoice = CosyVoice(args.model_dir, load_jit=False, load_trt=False, fp16=args.fp16)

k = 0
for name, transcript in [
    ("./asset/sqr3.wav", "我会把三段话切成3段，用来做"),
    ("./asset/wll3.wav", "声纹识别能力，多测一些"),
    ("./asset/wzy_stereo.wav", "明天是星期六啦，我要去上果力课啦，你们知道吗？"),
]:

    prompt_speech_16k = load_wav(name, 16000)
    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "我们是<strong>X robot</strong>小组，[laughter]，在做角色扮演的机器人。",
            transcript,
            prompt_speech_16k,
            stream=False,
        )
    ):
        audio_path = os.path.join(args.output_dir, "zero_shot_{}.wav".format(k))
        torchaudio.save(audio_path, j["tts_speech"], cosyvoice.sample_rate)
        k += 1

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
