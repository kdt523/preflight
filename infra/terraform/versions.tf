terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  # State lives in S3 with native locking. Configure per environment:
  #   terraform init -backend-config=env/staging.backend.hcl
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project     = "preflight"
      Environment = var.env
      ManagedBy   = "terraform"
    }
  }
}

data "aws_caller_identity" "me" {}
data "aws_availability_zones" "available" { state = "available" }

locals {
  name   = "preflight-${var.env}"
  azs    = slice(data.aws_availability_zones.available.names, 0, 2)
  image  = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  bucket = "${local.name}-${data.aws_caller_identity.me.account_id}"
}
