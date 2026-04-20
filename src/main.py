import os
import json
import pandas as pd
from dotenv import load_dotenv
import openai
from utils import transcrever_audio, gerar_voz

# 1. Configurações Iniciais
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def carregar_conhecimento_mercado():
    """Carrega as regras e dicas de trading da pasta data."""
    with open('data/conhecimento_mercado.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def carregar_perfil_investidor():
    """Carrega os dados pessoais do investidor do arquivo CSV."""
    df = pd.read_csv('data/perfil_investidor.csv')
    # Convertemos a primeira linha para um dicionário para facilitar o uso no prompt
    return df.iloc[0].to_dict()

def rodar_assistente():
    print("--- Iniciando UCL-Trade Assistant ---")
    
    # 2. Verificação do arquivo de áudio
    caminho_audio_input = "request_audio.wav"
    if not os.path.exists(caminho_audio_input):
        print(f"Erro: O arquivo {caminho_audio_input} não foi encontrado na raiz do projeto.")
        return

    # 3. Processamento STT (Voz para Texto)
    print("Transcrevendo áudio com Whisper...")
    try:
        texto_usuario = transcrever_audio(caminho_audio_input)
        print(f"Usuário disse: {texto_usuario}")
    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return

    # 4. Preparação do Contexto (Data)
    mercado = carregar_conhecimento_mercado()
    perfil = carregar_perfil_investidor()
    
    # Criamos um "System Prompt" robusto unindo os dois arquivos da pasta data
    prompt_sistema = (
        f"Você é o UCL-Trade, um assistente especializado em mercado financeiro. "
        f"Você está atendendo o investidor {perfil['nome']}, que tem perfil {perfil['perfil']} "
        f"e foca em {perfil['ativos_preferidos']}. "
        f"Use sua base de conhecimento para responder: {mercado}. "
        f"Seja direto, técnico e encorajador."
    )

    # 5. Inteligência Artificial (GPT-4)
    print("Consultando cérebro da IA...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": texto_usuario}
            ],
            temperature=0.7 # Equilíbrio entre precisão e naturalidade
        )
        
        resposta_final = response.choices[0].message.content
        print(f"UCL-Trade: {resposta_final}")
    except Exception as e:
        print(f"Erro na API da OpenAI: {e}")
        return

    # 6. Saída TTS (Texto para Voz)
    print("Gerando áudio de resposta...")
    caminho_audio_output = "response_audio.wav"
    gerar_voz(resposta_final, caminho_audio_output)
    
    print(f"Sucesso! Resposta salva em: {caminho_audio_output}")
    print("--- Operação Finalizada ---")

if __name__ == "__main__":
    rodar_assistente()
