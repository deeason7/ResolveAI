"""
Application configuration loaded from environment variables.

Pydantic BaseSettings reads from the process environment (and .env when
running locally). Inside Docker every value comes from env_file in
docker-compose.yml, so no .env file ships in the container image.
"""

from pydantic import Field
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
    # Managed Postgres (e.g. Neon) requires TLS; asyncpg ignores sslmode= in the
    # URL, so this flag adds connect_args={"ssl": True} on the engine instead.
    db_require_ssl: bool = False

    # Redis
    redis_url: str
    classification_queue: str = "classification:queue"
    resolution_queue: str = "resolution:queue"
    # A pending stream entry must be idle at least this long before another
    # consumer reclaims it (XAUTOCLAIM). The threshold is the safety margin that
    # keeps a message in-flight on a healthy worker — e.g. a multi-second LLM
    # call — from being stolen mid-flight; only a crashed worker's orphans, idle
    # past this, get swept up.
    reclaim_min_idle_ms: int = 60000

    # Worker poll cadence — the two knobs that bound idle Redis command volume.
    # Each loop spends one XAUTOCLAIM (PEL sweep) + one XREADGROUP (blocking
    # read): two commands per cycle. On a command-billed managed Redis (Upstash
    # free tier ~500K/month) the dev defaults idle at ~2 cmds / 5s ≈ 2M/month
    # per worker — several times over. The free deploy widens the block and
    # thins the sweep (e.g. 30000 / 4 ≈ 200K/month for both workers); the
    # defaults keep the local stack's snappy 5s, every-cycle loop.
    worker_block_ms: int = 5000
    worker_reclaim_every: int = Field(default=1, ge=1)  # PEL sweep every Nth cycle

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    # Managed Qdrant Cloud: set qdrant_url (e.g. https://xxxx.cloud.qdrant.io:6333)
    # + qdrant_api_key to use a hosted cluster. Empty url keeps the host/port path.
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Neo4j
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str
    # Managed Neo4j (Aura) names the db after the instance id, not "neo4j";
    # override with NEO4J_DATABASE. Community/Docker keep the single "neo4j".
    neo4j_database: str = "neo4j"

    # Ollama (primary classifier — fine-tuned Qwen2.5-3B served locally)
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "resolveai-sentiment"

    # Cloud LLM fallbacks (OpenAI-compatible)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""

    # LLM client behavior
    llm_timeout_s: float = 30.0
    classification_max_retries: int = 2
    # Cloud providers cap free-tier throughput (Groq: ~12K tokens/min). On a 429
    # the client honors the provider's Retry-After hint and re-tries the SAME
    # provider up to this many times before giving up to the fallback chain — so
    # a batch enqueue paces itself to the budget instead of fail-closing a chunk
    # of it into degraded "manual review" classifications.
    llm_rate_limit_retries: int = 5
    # Wait used only when a 429 carries no usable Retry-After header.
    llm_rate_limit_backoff_s: float = 10.0
    # Proactive tokens-per-minute pacing for the cloud provider. Groq's free
    # tier meters *tokens* per minute (~12K for llama-3.3-70b); a burst of
    # back-to-back completions trips it, and instructor swallows the 429 into a
    # retry exception before the reactive Retry-After backoff above can see it —
    # so the call fail-closes to the deterministic fallback. A token bucket
    # sized to this many tokens/min paces calls BEFORE they reach the provider.
    # 0 disables it: the local-first default needs no cap (Ollama has none), so
    # only the managed-tier deploy sets it (GROQ_TPM_LIMIT).
    groq_tpm_limit: int = 0
    # Drop the local Ollama provider and go straight to the cloud fallback.
    # Default off (the design is local-first); turn on for hardware that can't
    # serve the local SLM, so each request doesn't eat a guaranteed local
    # failure + wasted timeout before reaching the cloud.
    llm_skip_local: bool = False

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
