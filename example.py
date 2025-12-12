import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def cosyvoice_example():
    """ CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
    # sft usage
    print(cosyvoice.list_available_spks())
    # change stream=True for chunk stream inference
    for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
        torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # cross_lingual usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.',
                                                            './asset/cross_lingual_prompt.wav')):
        torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # vc usage
    for i, j in enumerate(cosyvoice.inference_vc('./asset/cross_lingual_prompt.wav', './asset/zero_shot_prompt.wav')):
        torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-Instruct')
    # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
    for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男',
                                                       'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>')):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice2_example():
    """ CosyVoice2 Usage, check https://funaudiollm.github.io/cosyvoice2/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # save zero_shot spk for future usage
    assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', 'my_zero_shot_spk') is True
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk')):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    cosyvoice.save_spkinfo()

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', './asset/zero_shot_prompt.wav')):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话<|endofprompt|>', './asset/zero_shot_prompt.wav')):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # bistream usage, you can use generator as input, this is useful when using text llm model as input
    # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
    def text_generator():
        yield '收到好友从远方寄来的生日礼物，'
        yield '那份意外的惊喜与深深的祝福'
        yield '让我心中充满了甜蜜的快乐，'
        yield '笑容如花儿般绽放。'
    for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_bistream_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def cosyvoice3_example():
    """ CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
    for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage, for supported control, check cosyvoice/utils/common.py#L28
    for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # hotfix usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
        torchaudio.save('hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice3_example()


if __name__ == '__main__':
    main()
