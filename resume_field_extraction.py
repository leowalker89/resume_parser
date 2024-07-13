from typing import List, Optional

from langchain.chains import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class CompanyOverview(BaseModel):
    """Information about the company offering the job."""
    about: Optional[str] = Field(..., description="Brief description of the company")
    mission_and_values: Optional[str] = Field(None, description="Company mission and values")
    size_and_locations: Optional[str] = Field(None, description="Company size and locations")

class RoleSummary(BaseModel):
    """Summary information about the job role."""
    title: str = Field(..., description="Job title")
    team_or_department: Optional[str] = Field(None, description="Team or department the role belongs to")
    role_type: Optional[str] = Field(..., description="Type of role (full-time, part-time, contract, etc.)")
    location: Optional[str] = Field(..., description="Location (on-site, remote, hybrid)")

class ResponsibilitiesAndQualifications(BaseModel):
    """Key responsibilities and qualifications for the job role."""
    responsibilities: List[str] = Field(..., description="Key responsibilities of the role")
    projects_and_problems: Optional[str] = Field(None, description="Types of projects and problems to be worked on")
    required_skills_and_experience: List[str] = Field(..., description="Required skills and experience for the role")
    preferred_skills_and_experience: Optional[List[str]] = Field(None, description="Preferred skills and experience for the role")

class CompensationAndBenefits(BaseModel):
    """Compensation and benefits offered for the job role."""
    salary_or_pay_range: Optional[str] = Field(None, description="Salary or hourly pay range")
    bonus_and_equity: Optional[str] = Field(None, description="Bonus and equity compensation")
    benefits: Optional[List[str]] = Field(None, description="Benefits (health insurance, retirement plans, PTO, etc.)")
    perks: Optional[List[str]] = Field(None, description="Perks (food, commuter benefits, learning stipend, etc.)")

class JobDescription(BaseModel):
    """Extracted information from a job description."""
    company_overview: CompanyOverview
    role_summary: RoleSummary
    responsibilities_and_qualifications: ResponsibilitiesAndQualifications
    compensation_and_benefits: CompensationAndBenefits