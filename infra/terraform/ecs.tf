resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = var.env != "prod"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_repository" "mlflow" {
  name                 = "${local.name}-mlflow"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.env != "prod"
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1, description = "keep last 30 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = "preflight.internal"
  vpc  = module.vpc.vpc_id
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30
}

locals {
  db_secret = aws_db_instance.db.master_user_secret[0].secret_arn

  common_env = [
    { name = "DB_HOST", value = aws_db_instance.db.address },
    { name = "DB_NAME", value = "preflight" },
    { name = "DB_USER", value = "preflight" },
    { name = "REDIS_URL", value = "rediss://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379/0" },
    { name = "S3_ENDPOINT_URL", value = "" },
    { name = "S3_BUCKET", value = aws_s3_bucket.data.bucket },
    { name = "AWS_REGION", value = var.region },
    { name = "MODEL_REGISTRY", value = "mlflow" },
    { name = "MLFLOW_TRACKING_URI", value = "http://mlflow.preflight.internal:5000" },
    { name = "SCORER_URL", value = "http://scorer.preflight.internal:8001" },
    { name = "MODEL_ROOT", value = "/tmp/models" },
  ]
  common_secrets = concat(
    [{ name = "DB_PASSWORD", valueFrom = "${local.db_secret}:password::" }],
    [for k, s in aws_secretsmanager_secret.app : { name = k, valueFrom = s.arn }],
  )

  # long-running services
  services = {
    ingest = { command = ["uvicorn", "apps.ingest.main:app", "--host", "0.0.0.0", "--port", "8000"], port = 8000, cpu = 256, memory = 512 }
    scorer = { command = ["uvicorn", "apps.scorer.main:app", "--host", "0.0.0.0", "--port", "8001"], port = 8001, cpu = 512, memory = 1024 }
    worker = { command = ["python", "-m", "apps.worker.main"], port = 9102, cpu = 512, memory = 1024 }
  }
  # one-off and scheduled jobs (same image, different command)
  jobs = {
    migrate     = { command = ["sh", "-c", "python scripts/ensure_db.py mlflow && alembic upgrade head"], cpu = 256, memory = 512 }
    materialize = { command = ["python", "scripts/materialize.py"], cpu = 1024, memory = 4096 }
    monitor     = { command = ["python", "scripts/monitor.py", "--auto-retrain"], cpu = 2048, memory = 8192 }
    backfill    = { command = ["python", "scripts/backfill.py"], cpu = 512, memory = 2048 }
    digest      = { command = ["python", "scripts/slack_digest.py"], cpu = 256, memory = 1024 }
  }
}

resource "aws_ecs_task_definition" "service" {
  for_each                 = local.services
  family                   = "${local.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name         = each.key
    image        = local.image
    command      = each.value.command
    essential    = true
    portMappings = [{ containerPort = each.value.port, name = each.key }]
    environment  = local.common_env
    secrets      = local.common_secrets
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = each.key
      }
    }
  }])
}

resource "aws_ecs_task_definition" "job" {
  for_each                 = local.jobs
  family                   = "${local.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name        = each.key
    image       = local.image
    command     = each.value.command
    essential   = true
    environment = local.common_env
    secrets     = local.common_secrets
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = each.key
      }
    }
  }])
}

resource "aws_service_discovery_service" "svc" {
  for_each = toset(["scorer", "mlflow"])
  name     = each.key
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.internal.id
    dns_records {
      type = "A"
      ttl  = 10
    }
  }
}

resource "aws_ecs_service" "svc" {
  for_each        = local.services
  name            = each.key
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.service[each.key].arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.task_subnets
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = local.task_public_ip
  }

  dynamic "load_balancer" {
    for_each = contains(["ingest", "scorer"], each.key) ? [1] : []
    content {
      target_group_arn = aws_lb_target_group.svc[each.key].arn
      container_name   = each.key
      container_port   = each.value.port
    }
  }

  dynamic "service_registries" {
    for_each = each.key == "scorer" ? [1] : []
    content { registry_arn = aws_service_discovery_service.svc["scorer"].arn }
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # a deploy that never gets healthy rolls itself back
  }

  lifecycle { ignore_changes = [desired_count] }
}

# --- MLflow tracking server + registry (internal only: its UI has no auth) -------------------
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${local.name}-mlflow"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name         = "mlflow"
    image        = "${aws_ecr_repository.mlflow.repository_url}:latest"
    essential    = true
    portMappings = [{ containerPort = 5000 }]
    environment = [
      { name = "DB_HOST", value = aws_db_instance.db.address },
      { name = "ARTIFACTS", value = "s3://${aws_s3_bucket.data.bucket}/mlflow" },
    ]
    secrets = [{ name = "DB_PASSWORD", valueFrom = "${local.db_secret}:password::" }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "mlflow"
      }
    }
  }])
}

resource "aws_ecs_service" "mlflow" {
  name            = "mlflow"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = local.task_subnets
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = local.task_public_ip
  }
  service_registries { registry_arn = aws_service_discovery_service.svc["mlflow"].arn }
}
