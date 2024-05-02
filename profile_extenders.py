from typing import List, Optional, Dict
from langchain_core.pydantic_v1 import BaseModel, Field

class BlogPost(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None

class Hackathon(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    achievements: Optional[List[str]] = None

class OtherInfo(BaseModel):
    category: Optional[str] = None
    description: Optional[str] = None