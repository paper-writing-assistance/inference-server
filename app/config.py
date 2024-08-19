from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PDF_SOURCE_DIR: Path = Path('.') / 'pdf'
    RESULT_DIR: Path = Path('.') / 'result'

    PARSER_MODEL_PATH: Path = Path()
    PARSER_CONFIG_PATH: Path = Path()
    LM_MODEL_PATH: Path = Path()

    

settings = Settings()
