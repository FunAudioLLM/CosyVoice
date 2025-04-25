import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import logging
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import argparse
import time
import numpy as np
from stream_player import StreamPlayer

# 设置根目录并添加第三方库路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))


# 设置日志级别为 DEBUG
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)

# 确保设置影响所有模块
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)


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
parser.add_argument(
    "--use_flow_cache", action="store_true", default=False, help="是否使用流式缓存"
)

args = parser.parse_args()

print(f"使用模型目录: {args.model_dir}")
cosyvoice = CosyVoice2(
    args.model_dir,
    load_jit=False,
    load_trt=True,
    fp16=args.fp16,
    use_flow_cache=args.use_flow_cache,
)

print(cosyvoice.list_available_spks())


# # prompt_speech_16k = load_wav("./asset/sqr3.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/wll3.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/wzy_read_poet_27s.wav", 16000)
# prompt_speech_16k = load_wav("./asset/harry_potter_snape_injured.wav", 16000)
# # prompt_speech_16k = load_wav("./asset/laoxu.wav", 16000)
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         # "声纹识别能力，多测一些",
#         # '明天是星期六啦，我要去上果粒课啦，你们知道吗？',
#         "I’m not hungry. That explains the blood. Listen. Last night, I'm guessing Snape let the troll in as a diversion, so he could get past that dog. But he got bit, that's why he's limping. The day I was at Gringotts, Hagrid took something out of the vault. Said it was Hogwarts business, very secret. That's what the dog's guarding. That's what Snape wants.  I never get mail.",
#         # "啊这个也能理解啊，因为七牛毕竟，是国内最早做云存储的公司。嗯，所以我想，就是和云存储相关的交流，可以在这个这个会之后自由讨论的时候，我们只管沟通啊。知无不言，言无不尽，哼哼。",
#         # "我最喜欢夏天，满地的鲜花，这里一朵，那里一朵， 真比天上的星星还多。 夜晚，我数着天上的星星，真比地上的花儿还要多。那里一颗，真比天上的花还，花儿还多。",
#         prompt_speech_16k,
#         stream=args.use_flow_cache,
#     )
# ):
#     torchaudio.save(
#         "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )

# # save zero_shot spk for future usage
# assert (
#     cosyvoice.add_zero_shot_spk(
#         # "声纹识别能力，多测一些", prompt_speech_16k, "wll"
#         # '明天是星期六啦，我要去上果粒课啦，你们知道吗？', prompt_speech_16k, "wzy"
#         # "啊这个也能理解啊，因为七牛毕竟，是国内最早做云存储的公司。嗯，所以我想，就是和云存储相关的交流，可以在这个这个会之后自由讨论的时候，我们只管沟通啊。知无不言，言无不尽，哼哼。", prompt_speech_16k, "laoxu"
#         # "我最喜欢夏天，满地的鲜花，这里一朵，那里一朵， 真比天上的星星还多。 夜晚，我数着天上的星星，真比地上的花儿还要多。那里一颗，真比天上的花还，花儿还多。",
#         "I’m not hungry. That explains the blood. Listen. Last night, I'm guessing Snape let the troll in as a diversion, so he could get past that dog. But he got bit, that's why he's limping. The day I was at Gringotts, Hagrid took something out of the vault. Said it was Hogwarts business, very secret. That's what the dog's guarding. That's what Snape wants.  I never get mail.",
#         prompt_speech_16k,
#         "hp",
#     )
#     is True
# )
# cosyvoice.save_spkinfo()


player = StreamPlayer(sample_rate=cosyvoice.sample_rate, channels=1, block_size=18048)
player.start()


print(
    "\n按回车使用默认文本，输入新文本后回车使用新文本，输入q后回车退出, 输入@后回车使用新指令\n"
)


while True:
    # 交互式循环，可以反复输入文本生成语音
    # speaker = "xiaoluo_mandarin"
    # speaker = "Donald J. Trump"
    # default_tts_text = "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. "
    default_speaker = "hp"
    default_tts_text = "从此每当害怕时，他就想起那个和伙伴共同编织星光的夜晚 [noise] ，勇气便像萤火虫般在心底亮起来。"
    default_instruct_text = "用很慢的语速读这个故事"
    speaker = default_speaker
    tts_text = default_tts_text
    instruct_text = default_instruct_text
    # 获取用户输入
    user_input = input(
        f"请输入文本 (格式: ' speaker @ tts_text @ instruct_text')  退出: q "
    )

    # 检查是否退出
    if user_input.strip() == "q":
        print("退出语音生成循环")
        break

    if len(user_input) > 1:
        speaker = user_input.split("@")[0]
    if len(user_input.split("@")) > 1:
        speaker = user_input.split("@")[0]
        tts_text = user_input.split("@")[1]
    if len(user_input.split("@")) > 2:
        speaker = user_input.split("@")[0]
        tts_text = user_input.split("@")[1]
        instruct_text = user_input.split("@")[2]

    print(f"SPEAKER 是： {speaker}， tts_text 是： {tts_text}")
    start_time = time.time()
    for i, j in enumerate(
        # cosyvoice.inference_instruct2(
        #     tts_text,
        #     instruct_text,
        #     prompt_speech_16k,
        #     stream=True,
        #     speed=0.8,
        #     text_frontend=True,
        # )
        cosyvoice.inference_sft(
            tts_text,
            speaker,
            stream=args.use_flow_cache,
        )
    ):
        current_time = time.time()
        # logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
        start_time = current_time

        # torchaudio.save(
        #     "sft_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        # )
        player.play(j["tts_speech"].numpy().T)

# 停止播放器
player.stop()

# # 最后一个示例，保存到文件而不是播放
# start_time = time.time()
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         # "这句话里面到底在使用了谁的语音呢？",
#         "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
#         "我会把三段话切成3段，用来做",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time
#     torchaudio.save(
#         "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )

# # instruct usage
# for i, j in enumerate(
#     cosyvoice.inference_instruct2(
#         "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
#         "用四川话说这句话",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     torchaudio.save("instruct_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate)

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

# start_time = time.time()

# for i, j in enumerate(
#     cosyvoice.inference_cross_lingual(
#         "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
#         "没有用到的文本",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     print(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time
#     torchaudio.save(
#         "fine_grained_control_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
#     )

# # 使用改进的播放机制进行流式语音生成和播放
# print("ATTENTION: 文本已经给到模型，开始生成语音啦！！！")
# start_time = time.time()

# # 流式生成并添加到播放队列
# for i, j in enumerate(
#     cosyvoice.inference_zero_shot(
#         # "这句话里面到底在使用了谁的语音呢？",
#         "CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。make america great again. ",
#         "我会把三段话切成3段，用来做",
#         prompt_speech_16k,
#         stream=True,
#     )
# ):
#     current_time = time.time()
#     logging.info(f"第 {i} 次生成耗时: {current_time - start_time:.2f} 秒")
#     start_time = current_time

#     player.play(j["tts_speech"].numpy().T)
