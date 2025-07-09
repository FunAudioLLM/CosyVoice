import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('example_sample/1_224p_16k_30s.wav', 16000)
prompt_speech_16k = load_wav('example_sample/chinese_soccer_commentary_16k_30s.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('比赛已经开始巴利控球,必须指出的是,他们曾在联赛首场比赛中交过手,当时切尔西以1比3获胜。巴利率先取得领先。需要提醒的是, 在那场比赛中他们也曾领先, 但最终切尔西轻松逆转。没错, 这正是我想说的。我清楚地记得那场比赛。巴利开局不错, 进了一球, 但切尔西最终轻松取胜。因此, 今天他们也不应该遇到太大的麻烦, 但正如你所说的, 足球比赛是复杂的。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('比赛已经开始, 巴利控球。必须指出的是, 他们曾在联赛首场比赛中交过手, 当时切尔西以1比3获胜。巴利率先取得领先。需要提醒的是, 在那场比赛中他们也曾领先, 但最终切尔西轻松逆转。没错, 这正是我想说的。我清楚地记得那场比赛。巴利开局不错, 进了一球, 但切尔西最终轻松取胜。因此, 今天他们也不应该遇到太大的麻烦, 但正如你所说的, 足球比赛是复杂的。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('比赛已经开始，巴利控球。必须指出的是，他们曾在联赛首场比赛中交过手，当时切尔西以1比3获胜。巴利率先取得领先。需要提醒的是，在那场比赛中他们也曾领先，但最终切尔西轻松逆转。没错，这正是我想说的。我清楚地记得那场比赛。巴利开局不错，进了一球，但切尔西最终轻松取胜。因此，今天他们也不应该遇到太大的麻烦，但正如你所说的，足球比赛是复杂的。', prompt_speech_16k, stream=False)):
    torchaudio.save('speech_generation_samples/same_lang/fine_grained_control_{}_0708.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('比赛已经开始，巴利控球。必须指出的是，他们曾在联赛首场比赛中交过手，当时切尔西以1比3获胜。巴利率先取得领先。需要提醒的是，在那场比赛中他们也曾领先，但最终切尔西轻松逆转。没错，这正是我想说的。我清楚地记得那场比赛。巴利开局不错，进了一球，但切尔西最终轻松取胜。因此，今天他们也不应该遇到太大的麻烦，但正如你所说的，足球比赛是复杂的。', '用激动的语气说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('speech_generation_samples/same_lang/instruct_{}_emotion.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物, '
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐, '
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)