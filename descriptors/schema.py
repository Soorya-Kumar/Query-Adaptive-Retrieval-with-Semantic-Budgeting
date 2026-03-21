from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RelativiyClass(str, Enum):
    recent = "recent"
    historical = "historical"
    timeless = "timeless"


class TagWithConfidence(BaseModel):
    tag: str
    confidence: float = Field(ge=0.0, le=1.0)


class ChunkDescriptor(BaseModel):
    fine: List[TagWithConfidence]
    mid: List[TagWithConfidence]
    coarse: List[TagWithConfidence]
    anchor_year: Optional[int] = None
    relativity_class: RelativiyClass