import json
import requests
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, 
    BackgroundTasks, 
    HTTPException,
    Response, 
    UploadFile, 
    status
)
from pydantic_core import ValidationError
from openai import OpenAI

from app.core.config import settings
from app.models import (
    UploadStatus,
    UpdateUploadStatus,
    Paper,
    PaperInference,
    PaperSummary
)
from app.utils import upload_file_to_s3


def load_model():
    """
    Load the model and return it
    """
    model = None
    return model


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model_name"] = load_model()
    yield


app = FastAPI(lifespan=lifespan)


def update_status(status_data: UpdateUploadStatus) -> UploadStatus:
    """
    Send status info to backend server
    """
    res = requests.put(
        "http://localhost:8000/api/v1/paper/status", 
        json=status_data.model_dump()
    ).json()
    return UploadStatus.model_validate(obj=res)


def detect_bounding_box(file: UploadFile):
    """
    Detect bounding box with LayoutParser
    """
    pass


def parse_metadata(detection) -> PaperInference:
    """
    Parse metadata with LayoutLMv3
    """
    return PaperInference(
        id="default-inference-id",
        abstract="Lorem ipsum ..."
    )


def extract_paper_summary(data: PaperInference) -> Paper:
    """
    Extract domain, problem, solution and keywords using OpenAI API
    """
    client = OpenAI()

    prompt = """You are an assistant for writing academic papers. You are 
    skilled at extracting research domain, problems of previous studies, 
    solution to the problem in single sentence and keywords. Your answer must 
    be in format of JSON {\"domain\": string, \"problem\": string, \"solution\"
    : string, \"keywords\": [string]}."""
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data.abstract}
        ]
    )
    res = completion.choices[0].message.content
    print(f"DEBUG:    OpenAI response {res}")

    try:
        summary = PaperSummary.model_validate(obj=json.loads(res))
        paper_data = Paper.model_validate(
            obj=(data.model_dump() | {"summary": summary}))
        return paper_data
    except ValidationError as e:
        print(e)
        return None


def async_infernce(file: UploadFile, upload_status: UploadStatus):
    ##### START INFERENCE #####
    parsed_metadata = parse_metadata(None)
    print(f"INFO:     Contents parsed for request {upload_status.request_id}")
    ###### END INFERENCE ######

    # Upload images to S3
    for img_file in []:
        upload_file_to_s3(img_file, "img")
    upload_status = update_status(
        UpdateUploadStatus(
            request_id=upload_status.request_id,
            images_uploaded=True
        )
    )
    print(f"INFO:     Images uploaded for request {upload_status.request_id}")

    # Extract summary using OpenAI API
    paper_data = extract_paper_summary(parsed_metadata)
    if not paper_data:
        return
    upload_status = update_status(
        UpdateUploadStatus(
            request_id=upload_status.request_id,
            keywords_extracted=True
        )
    )
    print(f"INFO:     Summary extracted for request {upload_status.request_id}")

    # Save to database
    paper = Paper.model_validate(obj=paper_data)
    res = requests.put(
        url=f"{settings.BACKEND_API_URI}/paper/create",
        json=paper.model_dump()
    ).json()
    upload_status = update_status(
        UpdateUploadStatus(
            request_id=upload_status.request_id,
            metadata_stored=True
        )
    )
    print(f"INFO:     Metadata stored for request {upload_status.request_id}")


@app.post("/predict")
def predict(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    res = requests.post(
        url=f"{settings.BACKEND_API_URI}/paper/status",
        json={ "filename": file.filename },
    ).json()
    upload_status = UploadStatus.model_validate(obj=res)

    # Upload PDF file to S3
    url = upload_file_to_s3(file, "pdf")
    if not url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file"
        )
    upload_status = update_status(
        UpdateUploadStatus(
            request_id=upload_status.request_id,
            pdf_uploaded=True
        )
    )
    
    background_tasks.add_task(async_infernce, file, upload_status)

    return Response(
        status_code=status.HTTP_202_ACCEPTED,
        content=f"Request {upload_status.request_id} added successfully. PDF file uploaded to: {url}"
    )


@app.get("/ping")
def ping():
    return "Pong"