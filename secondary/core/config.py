from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    BACKEND_API_URI: str = "http://localhost:8000/api/v1"

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_BUCKET_NAME: str


settings = Settings()