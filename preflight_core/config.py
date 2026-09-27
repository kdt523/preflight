from functools import lru_cache
from urllib.parse import quote

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: SecretStr = SecretStr("")
    github_api_url: str = "https://api.github.com"
    github_webhook_secret: SecretStr = SecretStr("")
    preflight_hash_salt: SecretStr = SecretStr("")

    database_url: str = "postgresql+psycopg://preflight:preflight@localhost:5433/preflight"
    # On AWS the URL is assembled from parts: the RDS-managed password arrives separately
    # from Secrets Manager and may contain characters that must be URL-encoded.
    db_host: str | None = None
    db_name: str = "preflight"
    db_user: str = "preflight"
    db_password: SecretStr | None = None
    redis_url: str = "redis://localhost:6379/0"
    events_stream: str = "gh:events"

    # GitHub App (live webhooks + Check Runs)
    github_app_id: str = ""
    github_app_private_key_path: str | None = None
    github_app_private_key: SecretStr | None = None  # PEM inline (Secrets Manager on AWS)

    # serving
    scorer_url: str = "http://localhost:8001"
    scorer_api_token: SecretStr | None = None  # required on /v1/* when set
    fail_open_timeout_s: float = 2.0
    model_root: str = "models"
    model_registry: str = "local"  # local | mlflow
    mlflow_model_name: str = "preflight"

    # notifications
    slack_webhook_url: SecretStr | None = None

    s3_endpoint_url: str | None = "http://localhost:8333"  # empty -> real AWS S3
    s3_bucket: str = "preflight-raw"
    aws_region: str = "ap-south-1"
    # Local only. On AWS these stay unset and boto3 uses the task's IAM role.
    aws_access_key_id: str | None = None
    aws_secret_access_key: SecretStr | None = None

    @model_validator(mode="after")
    def _derive(self) -> "Settings":
        if self.db_host and self.db_password:
            pw = quote(self.db_password.get_secret_value(), safe="")
            self.database_url = (
                f"postgresql+psycopg://{self.db_user}:{pw}@{self.db_host}:5432/{self.db_name}"
            )
        if not self.s3_endpoint_url:
            self.s3_endpoint_url = None
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
