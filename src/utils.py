import os
import whisper
from gtts import gTTS

def transcrever_audio(caminho_audio):
    model = whisper.load_model("small")
    result = model.transcribe(caminho_audio, fp16=False)
    return result["text"]

def gerar_voz(texto, caminho_saida):
    tts = gTTS(text=texto, lang='pt')
    tts.save(caminho_saida)
