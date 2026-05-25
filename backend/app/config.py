"""
Application configuration loaded from environment variables.

Pydantic BaseSettings reads from the process environment (and .env when
running locally). Inside Docker every value comes from env_file in
docker-compose.yml, so no .env file ships in the container image.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # PostgreSQL
    database_url: str

    # Redis
    redis_url: str

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # Neo4j
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Ollama
    ollama_base_url: str = "http://ollama:11434"

    # Cloud LLM fallbacks
    groq_api_key: str = ""
    openai_api_key: str = ""

    # AWS / Bedrock — boto3 picks up credentials from ~/.aws/ or env vars
    # automatically; we only need the region here. The two access-key fields
    # are read-and-forward: pydantic-settings ignores them unless explicitly
    # set, and boto3 sees them via os.environ regardless.
    aws_region: str = "us-east-1"

    # Auth
    jwt_secret_key: str
    jwt_refresh_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7


settings = Settings()
