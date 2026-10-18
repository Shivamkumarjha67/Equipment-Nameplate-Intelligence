from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class SpecificationModel(BaseModel):
    """Technical ratings stamped on an equipment nameplate."""
    voltage_v: Optional[int] = Field(None, description="Nominal operational voltage in Volts (e.g. 230, 400)")
    current_a: Optional[float] = Field(None, description="Rated operational current in Amperes")
    power_kw: Optional[float] = Field(None, description="Mechanical rated power in Kilowatts")
    frequency_hz: Optional[int] = Field(None, description="AC line frequency (50 Hz or 60 Hz)")
    rpm: Optional[int] = Field(None, description="Full-load rotational speed in RPM / min-1")


class EquipmentRecord(BaseModel):
    """Core industrial asset identity."""
    serial_number: str = Field(..., description="Unique equipment serial identifier (e.g. SN-XXXX-XXXX)")
    model: str = Field("UNKNOWN", description="Manufacturer model designation")
    manufacturer: str = Field(..., description="OEM manufacturer name (e.g. SIEMENS, ABB, WEG)")
    location_code: str = Field("BAY-01", description="Assigned plant location identifier")
    last_inspected: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of visual audit")
    specifications: Optional[SpecificationModel] = None


class FieldExtractionResult(BaseModel):
    """Result of an individual OCR field extraction."""
    value: Any = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class NameplateExtractionResult(BaseModel):
    """Full parameter matrix extracted from a nameplate."""
    manufacturer: FieldExtractionResult
    model: FieldExtractionResult
    serial_number: FieldExtractionResult
    voltage: FieldExtractionResult
    current: FieldExtractionResult
    power: FieldExtractionResult
    frequency: FieldExtractionResult
    rpm: FieldExtractionResult


class QualityGateTriage(BaseModel):
    """Machine learning quality gate evaluation."""
    triage_label: int = Field(..., description="0 = ACCEPTED, 1 = MANUAL_REVIEW_REQUIRED")
    status_label: str = Field(..., description="'ACCEPTED' or 'MANUAL_REVIEW_REQUIRED'")
    is_physically_plausible: bool = Field(..., description="True if electrical triplet aligns with power law")
    blur_score: float = Field(..., description="Laplacian focus variance")
    mean_ocr_confidence: float = Field(..., ge=0.0, le=1.0)
    power_discrepancy: float = Field(..., description="Relative difference between nominal and theoretical power")


class GraphRAGResponse(BaseModel):
    """Response model from the Graph-RAG query coordinator."""
    user_question: str
    cypher_query: str
    raw_records: list[dict[str, Any]]
    summary: str

