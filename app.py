# ============================================================
# Clinical Aggregator API
# API simples em Flask para extrair texto de notas clínicas
# Usa Google Gemini 1.5 Flash
# ============================================================

import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import base64

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa Flask
app = Flask(__name__)

# Habilita CORS (permite chamadas de qualquer origem)
# Em produção, você pode restringir para apenas liviamed.ai
CORS(app, origins=[
    "https://liviamed.ai",
    "https://docdoor-livia-web-hmv-ebg7ekg4epfkf5az.brazilsouth-01.azurewebsites.net",
    "http://localhost:3000",
    "http://localhost:5173",
    "*"  # Remove em produção se quiser restringir
])

# ============================================================
# CONFIGURAÇÃO DO GEMINI
# ============================================================

# A chave vem da variável de ambiente (configurada no Render)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Prompts para o Gemini
SYSTEM_INSTRUCTION = """
Você é um assistente de triagem médica especializado em processamento de documentos clínicos.

REGRAS DE SEGURANÇA (LGPD):
- NUNCA extraia dados pessoais identificáveis: nomes, CPF, RG, endereços, telefones, e-mails
- Substitua identificadores por termos genéricos
- Mantenha apenas dados clinicamente relevantes

FORMATO:
- Se o documento NÃO contiver informações clínicas: responda "NOT_CLINICAL: [motivo]"
- Se contiver: crie uma narrativa estruturada e concisa
"""

USER_PROMPT = """
Analise o documento anexado e extraia as informações clínicas relevantes.

Se houver conteúdo clínico, organize em:
- Idade e sexo (sem nome)
- Queixa principal
- História da doença atual
- Antecedentes / Comorbidades
- Medicações em uso
- Alergias
- Exame físico (achados relevantes)
- Exames complementares
- Hipótese diagnóstica
- Conduta

Use abreviações médicas padrão. Seja conciso e objetivo.
Se não houver conteúdo clínico (ex: fatura, documento administrativo), indique claramente.
"""

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def get_mime_type(filename: str) -> str:
    """Detecta o MIME type baseado na extensão do arquivo"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    mime_map = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        'bmp': 'image/bmp'
    }
    return mime_map.get(ext, 'application/octet-stream')


def validate_file(filename: str, file_size: int) -> tuple:
    """Valida o arquivo antes de processar"""
    # Extensões permitidas
    allowed = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'tiff', 'tif', 'bmp'}
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext not in allowed:
        return False, f"Formato .{ext} não suportado. Use: PDF, PNG, JPG, etc."
    
    # Limite de 20MB
    max_size = 20 * 1024 * 1024
    if file_size > max_size:
        return False, f"Arquivo muito grande ({file_size / 1024 / 1024:.1f}MB). Limite: 20MB"
    
    return True, ""


def process_with_gemini(file_bytes: bytes, mime_type: str) -> str:
    """Processa o documento com Gemini 1.5 Flash"""
    
    # Inicializa cliente Gemini
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Cria a parte do arquivo
    file_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type
    )
    
    # Configuração de geração (temperatura baixa para extração factual)
    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=2000
    )
    
    # Chama o Gemini
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[file_part, USER_PROMPT],
        config=config,
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    return response.text.strip()


# ============================================================
# ROTAS DA API
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Rota de verificação - mostra que a API está funcionando"""
    return jsonify({
        "status": "online",
        "service": "Clinical Aggregator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "Extrai texto de documento clínico",
            "GET /health": "Verifica status da API"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check - útil para monitoramento"""
    return jsonify({
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY)
    })


@app.route('/extract', methods=['POST', 'OPTIONS'])
def extract():
    """
    Endpoint principal - extrai texto de documento clínico
    
    Envie um arquivo via multipart/form-data com o campo 'file'
    
    Resposta:
    - success: true/false
    - message: texto extraído ou mensagem de erro
    - error: código do erro (se houver)
    """
    
    # Handle preflight CORS
    if request.method == 'OPTIONS':
        return '', 204
    
    # Verifica se Gemini está configurado
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY não configurada")
        return jsonify({
            "success": False,
            "error": "CONFIG_ERROR",
            "message": "API não configurada corretamente. Contate o administrador."
        }), 500
    
    # Verifica se arquivo foi enviado
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "NO_FILE",
            "message": "Nenhum arquivo enviado. Envie um arquivo no campo 'file'."
        }), 400
    
    file = request.files['file']
    
    # Verifica se tem nome
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "NO_FILE",
            "message": "Arquivo sem nome."
        }), 400
    
    try:
        # Lê o arquivo
        file_bytes = file.read()
        filename = file.filename
        
        logger.info(f"Processando arquivo: {filename} ({len(file_bytes)} bytes)")
        
        # Valida
        is_valid, error_msg = validate_file(filename, len(file_bytes))
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "INVALID_FILE",
                "message": error_msg
            }), 400
        
        # Detecta MIME type
        mime_type = get_mime_type(filename)
        
        # Processa com Gemini
        result_text = process_with_gemini(file_bytes, mime_type)
        
        # Verifica se é conteúdo não-clínico
        if result_text.upper().startswith("NOT_CLINICAL"):
            reason = result_text.split(":", 1)[-1].strip() if ":" in result_text else "Documento não clínico"
            return jsonify({
                "success": False,
                "error": "NOT_CLINICAL",
                "message": f"Documento não contém informações clínicas: {reason}"
            }), 200  # 200 porque processou corretamente
        
        # Sucesso!
        return jsonify({
            "success": True,
            "message": result_text,
            "metadata": {
                "filename": filename,
                "size_bytes": len(file_bytes),
                "mime_type": mime_type
            }
        }), 200
        
    except Exception as e:
        logger.exception(f"Erro ao processar: {e}")
        return jsonify({
            "success": False,
            "error": "PROCESSING_ERROR",
            "message": "Erro ao processar documento. Tente novamente."
        }), 500


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == '__main__':
    # Porta vem da variável de ambiente (Render define automaticamente)
    port = int(os.environ.get('PORT', 5000))
    
    # Debug apenas em desenvolvimento local
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Iniciando servidor na porta {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
