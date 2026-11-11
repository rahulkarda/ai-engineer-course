# Phase 6 · Week 07 · Day 4 — Terraform Fundamentals

> Objective: Learn Terraform's core concepts — providers, resources, state, variables, modules, plan/apply.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

**Terraform** is an Infrastructure as Code (IaC) tool that lets you declare cloud resources in HCL (HashiCorp Configuration Language) and converge the real world to that declaration. The key insight: infrastructure becomes version-controlled, reviewable code.

**Why Terraform over manual console clicks?**
- Reproducible: run the same config in dev, staging, prod — get identical infra.
- Reviewable: `terraform plan` shows exactly what will change before applying.
- Reversible: `terraform destroy` tears down cleanly; history in git.
- Composable: **modules** package reusable infra patterns.

### Core workflow

```
Write HCL  →  terraform init  →  terraform plan  →  terraform apply
                  │                    │                   │
           download providers    show diff of       create/update/delete
           init state backend    intended changes   real cloud resources
```

## How it works internally

Terraform maintains a **state file** (`terraform.tfstate`) that maps your HCL resources to real cloud resource IDs. On each `plan`:

1. Reads current HCL (desired state).
2. Reads state file (last known actual state).
3. Calls provider APIs to refresh actual state.
4. Computes diff: what needs to be created/updated/destroyed.

**State is the source of truth.** Never edit it manually. Store it remotely (S3, GCS, Terraform Cloud) so the team shares the same view.

### Resource lifecycle

| Symbol in plan | Meaning |
|---|---|
| `+` | Create |
| `~` | Update in-place |
| `-` | Destroy |
| `-/+` | Replace (destroy then create) |
| `<=` | Data source (read-only) |

Replace is dangerous — `-/+` on a database means data loss. Always read the plan carefully.

### Provider model

Providers are plugins that translate HCL into API calls. Each provider is published to the Terraform Registry.

```hcl
# providers are declared in required_providers
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"   # ~> means >=5.0, <6.0
    }
  }
}

provider "aws" {
  region = var.aws_region
  # Credentials from environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  # or from ~/.aws/credentials — never hardcode in HCL
}
```

## Worked example

### Project structure

```
infra/
├── main.tf          # resources
├── variables.tf     # input variable declarations
├── outputs.tf       # output value declarations
├── terraform.tf     # provider + backend config
└── modules/
    └── container-service/
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

### Variables and locals

```hcl
# variables.tf
variable "environment" {
  type        = string
  description = "Deployment environment (dev, staging, prod)"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging, or prod"
  }
}

variable "aws_region" {
  type    = string
  default = "eu-west-1"
}

variable "app_port" {
  type    = number
  default = 8080
}

variable "db_password" {
  type      = string
  sensitive = true  # masked in plan output and logs
  # Pass via: TF_VAR_db_password env var or terraform.tfvars (gitignored)
}
```

```hcl
# locals.tf (inside main.tf or separate)
locals {
  name_prefix = "${var.environment}-ai-service"
  common_tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = "ai-service"
  }
}
```

### Core resources: VPC, ECR, ECS

```hcl
# main.tf

# ── ECR repository for container images ──────────────────────────────────────
resource "aws_ecr_repository" "ai_service" {
  name                 = local.name_prefix
  image_tag_mutability = "IMMUTABLE"   # prevent tag overwrites

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

# ── IAM role for ECS task execution ──────────────────────────────────────────
resource "aws_iam_role" "ecs_task_exec" {
  name = "${local.name_prefix}-task-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_exec" {
  role       = aws_iam_role.ecs_task_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── ECS cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = local.name_prefix

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

# ── ECS task definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "ai_service" {
  family                   = local.name_prefix
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_exec.arn

  container_definitions = jsonencode([{
    name  = "ai-service"
    image = "${aws_ecr_repository.ai_service.repository_url}:latest"
    portMappings = [{
      containerPort = var.app_port
      protocol      = "tcp"
    }]
    environment = [
      { name = "ENVIRONMENT", value = var.environment }
    ]
    # Secrets from AWS Secrets Manager — not in plain text
    secrets = [
      {
        name      = "DB_PASSWORD"
        valueFrom = aws_secretsmanager_secret.db_password.arn
      }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${local.name_prefix}"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = local.common_tags
}

# ── Secrets Manager for DB password ──────────────────────────────────────────
resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${local.name_prefix}/db-password"
  recovery_window_in_days = 7
  tags                    = local.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.db_password
}
```

### Outputs

```hcl
# outputs.tf
output "ecr_repository_url" {
  description = "ECR repository URL for pushing images"
  value       = aws_ecr_repository.ai_service.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}
```

### Remote state backend (S3)

```hcl
# terraform.tf
terraform {
  required_version = ">= 1.6.0"

  backend "s3" {
    bucket         = "my-terraform-state"   # pre-created S3 bucket
    key            = "ai-service/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"   # DynamoDB for state locking
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
```

### Reusable module

```hcl
# modules/container-service/variables.tf
variable "name"        { type = string }
variable "image_url"   { type = string }
variable "cpu"         { type = number; default = 512 }
variable "memory"      { type = number; default = 1024 }
variable "environment" { type = string }

# modules/container-service/main.tf  (abbreviated)
resource "aws_ecs_service" "this" {
  name            = var.name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.environment == "prod" ? 3 : 1
  launch_type     = "FARGATE"
  # ...
}

# Calling the module from root main.tf:
module "ai_service" {
  source      = "./modules/container-service"
  name        = local.name_prefix
  image_url   = "${aws_ecr_repository.ai_service.repository_url}:${var.image_tag}"
  environment = var.environment
}
```

## Exercises

1. Write a `variables.tf` that declares `instance_count` (number, default 1), validates it is between 1 and 10, and marks `api_key` as sensitive.

2. What is the difference between `terraform taint` and `terraform import`? When would you use each?

3. The team has two developers. They both run `terraform apply` at the same time. What problem occurs without a DynamoDB lock table? How does the lock prevent it?

4. Write a `data` block that looks up the latest Amazon Linux 2023 AMI ID dynamically (rather than hardcoding the AMI ID). Why is a dynamic lookup important?

5. A resource has `lifecycle { prevent_destroy = true }`. The plan still shows `-/+` for it. What happens when you run `terraform apply`? How do you override it for a legitimate replacement?

(Solutions: ../solutions/week-07/day-4.md)

## Self-check

- [ ] I can write a Terraform config with variables, locals, resources, and outputs
- [ ] I understand what the state file contains and why it must be stored remotely
- [ ] I can interpret all symbols in `terraform plan` output
- [ ] I can create a reusable module and call it from root config
- [ ] I know how to pass secrets to Terraform without hardcoding them

## Supplementary links

- [Terraform documentation](https://developer.hashicorp.com/terraform/docs)
- [Terraform AWS provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Best practices: Terraform state](https://developer.hashicorp.com/terraform/language/state)
- [Terraform modules](https://developer.hashicorp.com/terraform/language/modules)
