# MLflow server with the Postgres + S3 drivers the official image leaves out
FROM ghcr.io/mlflow/mlflow:v3.1.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.40.0
# DB_HOST / DB_PASSWORD / ARTIFACTS come from the ECS task (or compose)
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://preflight:${DB_PASSWORD}@${DB_HOST}:5432/mlflow --artifacts-destination ${ARTIFACTS} --serve-artifacts"]
