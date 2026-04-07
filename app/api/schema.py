from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    state_override: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    messages: List[Dict[str, Any]]
    phase: str
    intent: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]

class DocStatus(BaseModel):
    filename: str
    status: str
    chunks_count: int
