import whisper

model = whisper.load_model("turbo")
result = model.transcribe("/root/CosyVoice/atuan2_postprocess.wav")
print(result["text"])