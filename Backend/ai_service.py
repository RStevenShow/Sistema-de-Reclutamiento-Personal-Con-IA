import fitz  # PyMuPDF
import requests
import math
import re

# Configuración de conexión al microservicio de IA en Google Colab
# IMPORTANTE: Actualizar esta URL cada vez que se reinicie el notebook en Colab.
COLAB_URL = "https://joannie-lacrimatory-donnetta.ngrok-free.dev"

def load_models():
    """Verifica la disponibilidad del servicio remoto."""
    try:
        print(f"Verificando conexión con servicio IA en: {COLAB_URL}...")
        response = requests.get(f"{COLAB_URL}/", timeout=5)
        if response.status_code == 200:
            print("Servicio de IA operativo y accesible.")
        else:
            print(f"Advertencia: El servicio respondió con estado {response.status_code}")
    except Exception as e:
        print(f"Error crítico: No se pudo establecer conexión con el servicio remoto. {e}")

def extract_text_from_pdf(pdf_bytes):
    """Extrae texto plano de un archivo PDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        return ""

def extract_email_from_text(text):
    """Extrae email usando Regex."""
    try:
        # Patrón estándar de email
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        match = re.search(email_pattern, text)
        if match:
            return match.group(0)
    except Exception:
        pass
    return None

def extract_phone_from_text(text):
    """Extrae teléfono (Soporta formatos de 8 dígitos seguidos y formatos internacionales)."""
    try:
        # 1. Buscamos formatos internacionales complejos (+505 1234 5678)
        matches = re.findall(r"\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}", text)
        for m in matches:
            if len(re.sub(r"\D", "", m)) >= 8: return m.strip()
            
        # 2. Si no, buscamos números simples de 8 dígitos (Ej: 85872276)
        # \b asegura que no sea parte de un número más largo
        simple_match = re.search(r"\b\d{8}\b", text)
        if simple_match:
            return simple_match.group(0)
            
    except Exception:
        pass
    return "No detectado"

def translate_text(text):
    """Traduce texto a inglés."""
    try:
        response = requests.post(f"{COLAB_URL}/translate", json={"text": text[:2500]})
        if response.status_code == 200:
            return response.json()["translation"]
    except Exception:
        pass
    return ""

def get_embedding(text):
    """Obtiene vector."""
    try:
        response = requests.post(f"{COLAB_URL}/vectorize", json={"text": text})
        if response.status_code == 200:
            return response.json()["vector"]
    except Exception:
        pass
    return []

def extract_keywords(text):
    """Obtiene keywords."""
    try:
        response = requests.post(f"{COLAB_URL}/keywords", json={"text": text[:3000]})
        if response.status_code == 200:
            return response.json()["keywords"]
    except Exception:
        pass
    return []

def generate_rationale(cv_text_en, offer_text_en):
    """Genera justificación lógica basada en coincidencias."""
    try:
        cv_keywords = extract_keywords(cv_text_en)
        offer_keywords = extract_keywords(offer_text_en)
        
        # Normalizamos a minúsculas
        cv_set = set([k.lower() for k in cv_keywords])
        offer_set = set([k.lower() for k in offer_keywords])
        
        common_skills = list(offer_set.intersection(cv_set))
        
        if not common_skills:
            return "El análisis semántico indica similitud contextual general, aunque no se detectaron coincidencias terminológicas directas."
            
        formatted_skills = [s.title() for s in common_skills[:6]]
        skills_string = ", ".join(formatted_skills)
        
        return f"Perfil compatible. Se detectaron coincidencias clave en competencias requeridas: {skills_string}. Esto valida la experiencia técnica solicitada."

    except Exception as e:
        print(f"Error en lógica de justificación: {e}")
        return "Análisis completado basado en similitud vectorial."

def calculate_similarity(vec1, vec2):
    """Matemática de vectores."""
    if not vec1 or not vec2: return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 * mag2 == 0: return 0.0
    return (dot / (mag1 * mag2)) * 100


def explain_match(cv_text, offer_text):
    """Solicita al servidor IA una explicación generativa entre CV y oferta."""
    try:
        response = requests.post(
            f"{COLAB_URL}/explain",
            json={"cv_text": cv_text, "offer_text": offer_text}
        )
        if response.status_code == 200:
            return response.json()["explanation"]
        else:
            return f"Error: el servidor respondió con estado {response.status_code}"
    except Exception as e:
        return f"Error crítico: {e}"