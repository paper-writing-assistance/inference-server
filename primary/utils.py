import boto3
import urllib
from fastapi import UploadFile

from primary.core.config import settings


s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)


def upload_file_to_s3(file: UploadFile, dir: str) -> str | None:
    """
    Uploads a file to an S3 bucket.

    Args:
        file: The file to be uploaded.
        dir: The directory in the S3 bucket where the file will be 
            uploaded.

    Returns:
        The URL of the uploaded file if the upload is successful, 
        otherwise None.
    """
    if not file:
        return None
    try:
        s3_client.upload_fileobj(
            Fileobj=file.file,
            Bucket=settings.S3_BUCKET_NAME,
            Key=f"{dir}/{file.filename}"
        )
    except:
        return None
    url = f"https://s3-ap-northeast-2.amazonaws.com/{settings.S3_BUCKET_NAME}/{urllib.parse.quote(f"{dir}/{file.filename}", safe="~()*!.'")}"
    return url