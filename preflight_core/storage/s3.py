"""Raw object store (S3 in prod, SeaweedFS locally). Everything is gzipped on the way in."""

import gzip
import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from preflight_core.config import get_settings


class RawStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=gzip.compress(data))
        return key

    def put_json(self, key: str, obj: Any) -> str:
        return self.put_bytes(key, json.dumps(obj, separators=(",", ":")).encode())

    def get_bytes(self, key: str) -> bytes:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return gzip.decompress(body)

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))


@lru_cache
def get_raw_store() -> RawStore:
    s = get_settings()
    client = boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=(
            s.aws_secret_access_key.get_secret_value() if s.aws_secret_access_key else None
        ),
    )
    store = RawStore(client, s.s3_bucket)
    store.ensure_bucket()
    return store
