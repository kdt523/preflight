resource "aws_lb" "main" {
  name               = local.name
  load_balancer_type = "application"
  subnets            = module.vpc.public_subnets
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "svc" {
  for_each    = { ingest = 8000, scorer = 8001 }
  name        = "${local.name}-${each.key}"
  port        = each.value
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"
  health_check {
    path                = "/healthz"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 15
  }
  deregistration_delay = 15
}

locals {
  https = var.certificate_arn != ""
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type = local.https ? "redirect" : "fixed-response"
    dynamic "redirect" {
      for_each = local.https ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
    dynamic "fixed_response" {
      for_each = local.https ? [] : [1]
      content {
        content_type = "text/plain"
        message_body = "preflight"
        status_code  = "404"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = local.https ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "preflight"
      status_code  = "404"
    }
  }
}

locals {
  listener_arn = local.https ? aws_lb_listener.https[0].arn : aws_lb_listener.http.arn
}

resource "aws_lb_listener_rule" "webhooks" {
  listener_arn = local.listener_arn
  priority     = 10
  condition {
    path_pattern { values = ["/webhooks/*"] }
  }
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.svc["ingest"].arn
  }
}

resource "aws_lb_listener_rule" "scorer" {
  listener_arn = local.listener_arn
  priority     = 20
  condition {
    path_pattern { values = ["/v1/*"] }
  }
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.svc["scorer"].arn
  }
}
