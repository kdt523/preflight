# --- S3: raw API pages, redacted log tails, datasets, MLflow artifacts -------------------
resource "aws_s3_bucket" "data" {
  bucket = local.bucket
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "expire-raw-api-pages"
    status = "Enabled"
    filter { prefix = "raw/" }
    expiration { days = 180 }
  }
  rule {
    id     = "old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 30 }
  }
}

# --- RDS Postgres ------------------------------------------------------------------------
resource "aws_db_subnet_group" "db" {
  name       = local.name
  subnet_ids = module.vpc.private_subnets
}

resource "aws_db_instance" "db" {
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "16"
  instance_class               = var.db_instance_class
  allocated_storage            = 20
  max_allocated_storage        = 100
  storage_encrypted            = true
  db_name                      = "preflight"
  username                     = "preflight"
  manage_master_user_password  = true # password lives only in Secrets Manager
  db_subnet_group_name         = aws_db_subnet_group.db.name
  vpc_security_group_ids       = [aws_security_group.data.id]
  backup_retention_period      = var.env == "prod" ? 7 : 1
  deletion_protection          = var.env == "prod"
  skip_final_snapshot          = var.env != "prod"
  final_snapshot_identifier    = var.env == "prod" ? "${local.name}-final" : null
  performance_insights_enabled = false
}

# --- ElastiCache Redis (Streams + online store) --------------------------------------------
resource "aws_elasticache_subnet_group" "redis" {
  name       = local.name
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = local.name
  description                = "Preflight event stream and online feature store"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = var.redis_node_type
  num_cache_clusters         = 1
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.data.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  snapshot_retention_limit   = var.env == "prod" ? 3 : 0
}
