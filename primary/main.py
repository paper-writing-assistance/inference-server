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

from primary.core.config import settings
from primary.models import (
    UploadStatus,
    UpdateUploadStatus,
    Paper,
    PaperInference,
    PaperSummary
)
from primary.predict import predict
from primary.utils import upload_file_to_s3


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


app = FastAPI(
    lifespan=lifespan,
    title="I. Inference Server"
)


def update_status(status_data: UpdateUploadStatus) -> UploadStatus:
    """
    Send status info to backend server
    """
    res = requests.put(
        "http://localhost:8000/api/v1/paper/status", 
        json=status_data.model_dump()
    ).json()
    return UploadStatus.model_validate(obj=res)


async def async_infernce(file: UploadFile, upload_status: UploadStatus):
    # Inference
    paper_data = predict()
    print(f"INFO:     Contents parsed for request {upload_status.request_id}")

    # Upload images to S3
    for img_file in []:
        upload_file_to_s3(img_file, "img")
    upload_status = update_status(
        UpdateUploadStatus(
            request_id=upload_status.request_id,
            images_uploaded=True
        )
    )
    # print(f"INFO:     Images uploaded for request {upload_status.request_id}")

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

    # Pass files to secondary server
    res = requests.post(
        url=f"{settings.SECONDARY_INFERENCE_SERVER_URI}/summarize",
        files=None
    )
    print(f"INFO:     Requesting summary for {upload_status.request_id}")
    print(f"INFO:     {res.json()}")


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
    with open(file.filename, 'wb') as f:
        f.write(file.file.read())
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