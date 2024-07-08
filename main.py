import io,time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# sft usage
print(cosyvoice.list_avaliable_spks())
app = FastAPI()

@app.get("/api/voice/tts")
async def tts(query: str, role: str):
    start = time.process_time()
    output = cosyvoice.inference_sft(query, role)
    end = time.process_time()
    print("infer time:", end-start, "seconds")
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

@app.get("/api/voice/roles")
async def roles():
    return {"roles": cosyvoice.list_avaliable_spks()}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang=zh-cn>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            Get the supported tones from the Roles API first, then enter the tones and textual content in the TTS API for synthesis. <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """
