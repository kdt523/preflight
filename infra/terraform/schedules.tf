# Nightly online-store rebuild and hourly drift/performance monitor (which retrains when
# drift, a live PR-AUC drop, or the weekly schedule calls for it).
resource "aws_iam_role" "scheduler" {
  name = "${local.name}-scheduler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "scheduler.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = "ecs:RunTask", Resource = [for t in aws_ecs_task_definition.job : "${t.arn_without_revision}:*"] },
      { Effect = "Allow", Action = "iam:PassRole", Resource = [aws_iam_role.execution.arn, aws_iam_role.task.arn] },
    ]
  })
}

locals {
  schedules = {
    materialize = "cron(0 2 * * ? *)"   # 02:00 UTC nightly
    monitor     = "cron(15 * * * ? *)"  # hourly
    digest      = "cron(0 4 ? * MON *)" # Mondays 04:00 UTC (09:30 IST)
  }
}

resource "aws_scheduler_schedule" "job" {
  for_each                     = local.schedules
  name                         = "${local.name}-${each.key}"
  schedule_expression          = each.value
  schedule_expression_timezone = "UTC"
  flexible_time_window { mode = "OFF" }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.job[each.key].arn_without_revision
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = local.task_subnets
        security_groups  = [aws_security_group.tasks.id]
        assign_public_ip = local.task_public_ip
      }
    }
    retry_policy { maximum_retry_attempts = 1 }
  }
}
