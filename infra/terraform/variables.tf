variable "env" {
  description = "staging | prod"
  type        = string
  validation {
    condition     = contains(["staging", "prod"], var.env)
    error_message = "env must be staging or prod"
  }
}

variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "github_repo" {
  description = "owner/name allowed to deploy via GitHub Actions OIDC"
  type        = string
  default     = "kdt523/preflight"
}

variable "image_tag" {
  description = "Image tag to run (the deploy workflow passes the git SHA)"
  type        = string
  default     = "latest"
}

variable "use_nat_gateway" {
  description = "true: tasks in private subnets behind a NAT (~$32/mo). false: public subnets + public IPs, inbound only from the ALB."
  type        = bool
  default     = false
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS on the ALB. Empty = HTTP only (fine for a demo, webhooks are HMAC-signed)."
  type        = string
  default     = ""
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "redis_node_type" {
  type    = string
  default = "cache.t4g.micro"
}

variable "desired_count" {
  description = "Tasks per long-running service"
  type        = number
  default     = 1
}

variable "alert_email" {
  description = "Receives CloudWatch alarms (optional)"
  type        = string
  default     = ""
}
