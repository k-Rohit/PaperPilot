import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class OpenAIEmbeddingsSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENAI_EMBEDDINGS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    # Read OPENAI_API_KEY directly (no prefix) — that's the standard env var name.
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPENAI_EMBEDDINGS__API_KEY"),
    )
    model: str = "text-embedding-3-small"
    dimensions: int = 1024
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: float = 30.0


class OpenAIClientSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENAI_CLIENT__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPENAI_CLIENT__API_KEY"),
    )
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 1.0
    timeout_seconds: float = 30.0
    max_retries: int = 3


class DeepEvalSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="DEEPEVAL__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    synthesizer_model: str = "gpt-4o-mini"
    top_k: int = 5  # chunks retrieved per golden; keep equal to AskRequest.top_k so evals match the app
    max_concurrent: int = 5  # goldens answered in parallel when building the evaluation dataset
    retriever: Literal["pinecone"] = "pinecone"


class PineconeSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PINECONE_",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    api_key: str = ""
    index_name: str = "arxiv-papers"
    sparse_model: str = "pinecone-sparse-english-v0"
    embed_batch_size: int = 96  # max inputs per pc.inference.embed call for the sparse model
    dimension: int = 1024  # must equal OpenAIEmbeddingsSettings.dimensions, or every upsert is rejected
    cloud: str = "aws"
    region: str = "us-east-1"


class RedisSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

    # Cache settings
    ttl_hours: int = 6  # Cache TTL in hours


class Settings(BaseConfigSettings):
    postgres_database_url: str = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
    checkpoint_database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"  # adding a new variable for checkpoint database url as langgraph requires a different format
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 5  # kept small: Supabase's free-tier pooler allows few connections
    postgres_max_overflow: int = 0

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    openai_embeddings: OpenAIEmbeddingsSettings = Field(default_factory=OpenAIEmbeddingsSettings)
    openai_client: OpenAIClientSettings = Field(default_factory=OpenAIClientSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    deepeval: DeepEvalSettings = Field(default_factory=DeepEvalSettings)
    pinecone_: PineconeSettings = Field(default_factory=PineconeSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
