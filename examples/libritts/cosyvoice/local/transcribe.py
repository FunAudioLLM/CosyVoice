import sys
import time
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

def transcribe_audio_faster_whisper(audio_path):
    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = [segment.text.strip() for segment in segments]
    text = ". ".join(text)
    return text

if __name__ == "__main__":
    st = time.perf_counter()
    for _ in range(10):
        print(transcribe_audio_faster_whisper(sys.argv[1]))
    et = time.perf_counter()
    print(f"Elapsed time mean: {(et - st)/10:.2f}s")