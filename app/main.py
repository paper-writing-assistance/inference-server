from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Response, 
    UploadFile,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from model.processor import DocumentProcessor


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def root():
    return "pong"


# =========================================================
# Models
# =========================================================
class SemanticID(BaseModel):
    semantic_id: str


# =========================================================
# Inference API
# =========================================================
@app.post("/inference")
async def inference(
    file: UploadFile,
    id: SemanticID = Depends(),
    background_tasks: BackgroundTasks = None
):
    # Save source PDF file
    with open(settings.PDF_SOURCE_DIR / file.filename, "wb") as f:
        f.write(file.file.read())

    # Async inference
    background_tasks.add_task(async_infernce, file, id.semantic_id)

    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        content=f"Request added successfully"
    )


# =========================================================
# Background Task
# =========================================================
async def async_infernce(
    file: UploadFile, 
    semantic_id: str, 
):
    processor = DocumentProcessor(
        pdf_path=settings.PDF_SOURCE_DIR / file.filename,
        parser_model_path=settings.PARSER_MODEL_PATH,
        parser_config_path=settings.PARSER_CONFIG_PATH,
        lm_model_path=settings.LM_MODEL_PATH,
        result_path=settings.RESULT_DIR
    )

    result = processor.process()

    ######## More to come......
    