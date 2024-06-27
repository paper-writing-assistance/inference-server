from datetime import datetime

from pydantic import BaseModel


class UploadStatus(BaseModel):
    request_id: int
    filename: str
    requested_date: datetime
    pdf_uploaded: bool = False
    bbox_detected: bool = False
    metadata_parsed: bool = False
    images_uploaded: bool = False
    keywords_extracted: bool = False
    metadata_stored: bool = False


class UpdateUploadStatus(BaseModel):
    request_id: int
    pdf_uploaded: bool | None = None
    bbox_detected: bool | None = None
    metadata_parsed: bool | None = None
    images_uploaded: bool | None = None
    keywords_extracted: bool | None = None
    metadata_stored: bool | None = None


class PaperBody(BaseModel):
    paragraph_id: int
    section: str
    text: str


class PaperFigure(BaseModel):
    idx: int
    name: str
    caption: str
    related: list[int]
    summary: str


class PaperQuery(BaseModel):
    domain: str
    problem: str
    solution: str


class PaperSummary(PaperQuery):
    keywords: list[str]


class PaperBase(BaseModel):
    id: str
    title: str | None = None


class PaperInference(PaperBase):
    abstract: str | None= None
    body: list[PaperBody] | None = None
    impact: int | None = None
    published_year: str | None = None
    reference: list[str] | None = None
    figures: list[PaperFigure] | None = None
    tables: list[PaperFigure] | None = None
    authors: list[str] | None = None


class Paper(PaperInference):
    summary: PaperSummary | None = None
