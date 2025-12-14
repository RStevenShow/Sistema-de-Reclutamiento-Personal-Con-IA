from typing import Optional, List
from sqlmodel import SQLModel, Field, Column, Relationship
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT

# --- Modelos Base (Campos Comunes) ---

class JobOfferBase(SQLModel):
    title: str = Field(index=True)
    description_original: str
    # Campos para enriquecimiento de contexto
    salary_range: Optional[str] = None
    experience_years: Optional[int] = None
    skills_required: Optional[str] = None
    responsibilities: Optional[str] = None

class CandidateBase(SQLModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    file_path: Optional[str] = None

# --- Modelos de Lectura (DTOs de Salida) ---

class CandidateRead(CandidateBase):
    id: int
    match_score: float
    rationale: Optional[str] = None
    text_extracted: Optional[str] = None

class JobOfferRead(JobOfferBase):
    id: int
    description_en: Optional[str] = None
    vector: Optional[List[float]] = None
    # Relación anidada para devolver candidatos en la respuesta de la oferta
    candidates: List[CandidateRead] = []

class JobOfferCreate(JobOfferBase):
    pass

# --- Tablas de Base de Datos (Persistencia) ---

class JobOffer(JobOfferBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description_en: Optional[str] = None
    # Almacenamiento de vectores dimensionales
    vector: Optional[List[float]] = Field(sa_column=Column(ARRAY(FLOAT)))
    
    # Relación uno a muchos
    candidates: List["Candidate"] = Relationship(back_populates="job_offer")

class Candidate(CandidateBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Datos procesados y metadatos
    text_extracted: Optional[str] = None
    text_en: Optional[str] = None
    vector: Optional[List[float]] = Field(sa_column=Column(ARRAY(FLOAT)))
    
    # Resultados del análisis
    match_score: float = 0.0
    rationale: Optional[str] = None
    
    # Clave foránea
    job_offer_id: Optional[int] = Field(default=None, foreign_key="joboffer.id")
    job_offer: Optional[JobOffer] = Relationship(back_populates="candidates")