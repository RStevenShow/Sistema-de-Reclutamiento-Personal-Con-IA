import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select
from typing import List
from database import create_db_and_tables, get_session
from Models import JobOffer, JobOfferCreate, JobOfferRead, Candidate, CandidateRead
from ai_service import (
    load_models,
    translate_text,
    get_embedding,
    extract_text_from_pdf,
    calculate_similarity,
    generate_rationale,
    extract_email_from_text,
    extract_phone_from_text
)

app = FastAPI()

# Configuración de CORS para permitir la comunicación entre el frontend y el backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración del directorio de almacenamiento para archivos PDF
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Montaje de la carpeta de archivos estáticos para permitir la descarga de CVs
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.on_event("startup")
def on_startup():
    """Inicializa la base de datos y verifica la conectividad con el servicio de IA."""
    print("Iniciando sistema...")
    create_db_and_tables()
    load_models()

# --- Gestión de Ofertas ---

@app.post("/offers/", response_model=JobOfferRead)
def create_offer(offer: JobOfferCreate, session: Session = Depends(get_session)):
    """
    Crea una nueva oferta laboral.
    Concatena los campos descriptivos (título, descripción, requisitos, etc.)
    para generar un vector semántico enriquecido que mejore la precisión del matching.
    """
    print(f"Procesando nueva oferta: {offer.title}")
    
    # Construcción del contexto unificado para el modelo de embedding
    full_context = f"""
    Puesto: {offer.title}.
    Descripción: {offer.description_original}.
    Skills requeridos: {offer.skills_required or 'N/A'}.
    Responsabilidades: {offer.responsibilities or 'N/A'}.
    Experiencia mínima: {offer.experience_years or 0} años.
    Rango Salarial: {offer.salary_range or 'N/A'}.
    """
    
    # Procesamiento NLP: Traducción y generación de vector
    desc_en = translate_text(full_context)
    vector = get_embedding(desc_en)
    
    # Creación y persistencia del objeto JobOffer
    new_offer = JobOffer(
        title=offer.title,
        description_original=offer.description_original,
        salary_range=offer.salary_range,
        experience_years=offer.experience_years,
        skills_required=offer.skills_required,
        responsibilities=offer.responsibilities,
        description_en=desc_en,
        vector=vector
    )
    
    session.add(new_offer)
    session.commit()
    session.refresh(new_offer)
    return new_offer

@app.get("/offers/", response_model=List[JobOfferRead])
def read_offers(session: Session = Depends(get_session)):
    """Retorna el listado de todas las ofertas registradas."""
    return session.exec(select(JobOffer)).all()

@app.get("/offers/{offer_id}", response_model=JobOfferRead)
def read_offer(offer_id: int, session: Session = Depends(get_session)):
    """Retorna el detalle de una oferta específica incluyendo sus candidatos."""
    offer = session.get(JobOffer, offer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    return offer

# --- Procesamiento de Candidatos ---

@app.post("/offers/{offer_id}/upload_cvs", response_model=List[CandidateRead])
async def upload_cvs(offer_id: int, files: List[UploadFile] = File(...), session: Session = Depends(get_session)):
    """
    Endpoint principal de procesamiento de candidatos.
    1. Guarda el archivo PDF en el servidor.
    2. Extrae texto y metadatos (email, teléfono).
    3. Ejecuta análisis semántico (match) y generativo (justificación).
    4. Guarda el resultado en la base de datos.
    """
    offer = session.get(JobOffer, offer_id)
    if not offer:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    
    results = []
    
    for file in files:
        print(f"Analizando archivo: {file.filename}")
        
        # 1. Almacenamiento del archivo
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        content = await file.read()
        
        with open(file_location, "wb") as f:
            f.write(content)
            
        # 2. Extracción de datos crudos y metadatos
        text_es = extract_text_from_pdf(content)
        email = extract_email_from_text(text_es)
        phone = extract_phone_from_text(text_es)
        
        # 3. Procesamiento con IA (Traducción y Vectorización)
        text_en = translate_text(text_es)
        vec_cv = get_embedding(text_en)
        
        # 4. Cálculo de Similitud (Ranking)
        score = calculate_similarity(vec_cv, offer.vector)
        
        # 5. Generación de Justificación (LLM Generativo)
        rationale = generate_rationale(text_en, offer.description_en)
        
        # 6. Generación de URL pública para descarga
        # Nota: Asegurarse que el host coincida con el entorno de despliegue
        download_url = f"http://127.0.0.1:8000/static/{file.filename}"
        
        # 7. Persistencia
        cand = Candidate(
            name=file.filename,
            email=email,
            phone=phone,
            file_path=download_url,
            text_extracted=text_es,
            text_en=text_en,
            vector=vec_cv,
            match_score=score,
            rationale=rationale,
            job_offer_id=offer.id
        )
        
        session.add(cand)
        results.append(cand)
    
    session.commit()
    
    # Ordenamiento por puntuación descendente (Mejores candidatos primero)
    results.sort(key=lambda x: x.match_score, reverse=True)
    
    return results