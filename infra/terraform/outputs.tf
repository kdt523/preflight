output "webhook_url" {
  description = "Set this as the GitHub App webhook URL"
  value       = "${local.https ? "https" : "http"}://${aws_lb.main.dns_name}/webhooks/github"
}

output "scorer_url" {
  description = "PREFLIGHT_URL for the CLI and pytest plugin"
  value       = "${local.https ? "https" : "http"}://${aws_lb.main.dns_name}"
}

output "ecr_repository" { value = aws_ecr_repository.app.repository_url }
output "ecr_mlflow_repository" { value = aws_ecr_repository.mlflow.repository_url }
output "cluster" { value = aws_ecs_cluster.main.name }
output "deploy_role_arn" {
  description = "GitHub Actions secret AWS_DEPLOY_ROLE_ARN"
  value       = aws_iam_role.deploy.arn
}
output "task_subnets" { value = local.task_subnets }
output "task_security_group" { value = aws_security_group.tasks.id }
output "bucket" { value = aws_s3_bucket.data.bucket }
output "secret_ids" { value = { for k, s in aws_secretsmanager_secret.app : k => s.name } }
