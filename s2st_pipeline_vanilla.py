
import hydra
import librosa
from omegaconf import DictConfig, OmegaConf

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


@hydra.main(version_base="1.3", config_path="./configs", config_name="s2st_pipeline_vanilla.yaml")
def main(cfg: DictConfig):
    
    # Model Loading
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ## Load Whisper ASR model
    asr_processor = WhisperProcessor.from_pretrained(cfg.asr.model)
    asr_model = WhisperForConditionalGeneration.from_pretrained(cfg.asr.model).to(device)
    ## Load Language Translation model
    nmt_tokenizer = AutoTokenizer.from_pretrained(cfg.nmt.model)
    nmt_model = AutoModelForCausalLM.from_pretrained(
        cfg.nmt.model,
        torch_dtype="auto",
        device_map="auto"
    ).to(device)
    # Load Text to Speech model
    cosyvoice = CosyVoice2(cfg.tts.model, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # Data Loading
    ## Load an example audio file
    audio_path = "example_sample/1_224p_16k_30s.wav"
    audio, sr = librosa.load(audio_path, sr=16000)

    # Transcribe
    input_features = asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = asr_model.generate(input_features.to(device))
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Translation
    ## Prepare the model input
    prompt = "Translate the following text to Chinese: " + transcription[0]
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = nmt_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = nmt_tokenizer([text], return_tensors="pt").to(nmt_model.device)
    ## Conduct text completion
    generated_ids = nmt_model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    ## Parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = nmt_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = nmt_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # print("thinking content:", thinking_content)
    # print("content:", content)
    content = "check this sentence"
    # content = f"{content}。 {content_example}"

    # Generate speech
    tts_audio_path = "example_sample/chinese_soccer_commentary_16k_30s.wav"
    prompt_speech_16k = load_wav(tts_audio_path, 16000)
    # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # # save zero_shot spk for future usage
    # assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
    # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    # cosyvoice.save_spkinfo()

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    for i, j in enumerate(cosyvoice.inference_cross_lingual(content, prompt_speech_16k, stream=False)):
        torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage
    for i, j in enumerate(cosyvoice.inference_instruct2(content, '激动得说这句话', prompt_speech_16k, stream=False)):
        torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

    # bistream usage, you can use generator as input, this is useful when using text llm model as input
    # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
    # def text_generator():
    #     yield '收到好友从远方寄来的生日礼物，'
    #     yield '那份意外的惊喜与深深的祝福'
    #     yield '让我心中充满了甜蜜的快乐，'
    #     yield '笑容如花儿般绽放。'
    # for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


if __name__ == "__main__":
    main()