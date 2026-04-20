import os
import json
from dotenv import load_dotenv
import openai
from utils import transcrever_audio, gerar_voz

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def carregar_contexto():
    with open('data/conhecimento_mercado.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def rodar_assistente():
    # 1. Simulação de entrada (pegando o áudio gravado)
    texto_usuario = transcrever_audio("request_audio.wav")
    print(f"Usuário: {texto_usuario}")

    # 2. Adicionando o contexto do "Banco de Dados"
    contexto = carregar_contexto()
    prompt_sistema = f"Você é o UCL-Trade. Use estes dados para responder: {contexto}"

    # 3. Chamada para o GPT
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_usuario}
        ]
    )
    
    resposta_final = response.choices[0].message.content
    print(f"Assistente: {resposta_final}")

    # 4. Saída em Voz
    gerar_voz(resposta_final, "response.wav")

if __name__ == "__main__":
    rodar_assistente()
