# Secret *containers* only. Values are set out of band so they never enter Terraform state:
#   aws secretsmanager put-secret-value --secret-id preflight-staging/github-webhook-secret \
#       --secret-string '...'
locals {
  secret_names = {
    GITHUB_WEBHOOK_SECRET  = "github-webhook-secret"
    GITHUB_APP_ID          = "github-app-id"
    GITHUB_APP_PRIVATE_KEY = "github-app-private-key"
    GITHUB_TOKEN           = "github-token"
    PREFLIGHT_HASH_SALT    = "hash-salt"
    SCORER_API_TOKEN       = "scorer-api-token"
    SLACK_WEBHOOK_URL      = "slack-webhook-url"
  }
}

resource "aws_secretsmanager_secret" "app" {
  for_each                = local.secret_names
  name                    = "${local.name}/${each.value}"
  recovery_window_in_days = var.env == "prod" ? 30 : 0
}
