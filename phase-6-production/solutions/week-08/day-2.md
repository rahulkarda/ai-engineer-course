# Week 8, Day 2 — Managed Container Services: Solution Key

## Exercise 1: ECS Task Definition JSON for an AI API Container

Rules applied: no inline secrets, environment variables reference SSM Parameter Store or Secrets Manager ARNs, IAM role separation enforced.

```json
{
  "family": "ai-api-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ai-api-execution-role",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ai-api-task-role",
  "containerDefinitions": [
    {
      "name": "ai-api",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/ai-api:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "APP_ENV",
          "value": "production"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "REDIS_HOST",
          "value": "ai-cache.abc123.0001.use1.cache.amazonaws.com"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/ai-api/anthropic-key"
        },
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:ssm:us-east-1:123456789012:parameter/prod/ai-api/db-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "readonlyRootFilesystem": true,
      "linuxParameters": {
        "initProcessEnabled": true
      }
    }
  ]
}
```

**Why these choices**:
- `secrets` block (not `environment`) for credentials: ECS injects them at runtime by calling Secrets Manager / SSM. The value never appears in the task definition JSON stored in AWS, CloudTrail, or logs.
- `executionRoleArn` separate from `taskRoleArn` — see Exercise 2.
- `readonlyRootFilesystem: true` — defense-in-depth; forces any writable paths to be explicit volumes.
- `healthCheck.startPeriod: 60` — AI API containers often load models at startup; gives them time before ECS starts failing health checks and replacing tasks.
- `awsvpc` network mode — each task gets its own ENI and security group. Required for Fargate.

---

## Exercise 2: ECS Task Role vs Execution Role

These are two distinct IAM roles with completely different purposes.

### Execution Role (`executionRoleArn`)

**Who uses it**: The ECS agent (the AWS-managed daemon running on the node or Fargate infrastructure), not your application code.

**Purpose**: Allows ECS to *set up* the task — pull the container image from ECR, retrieve secrets from Secrets Manager / SSM to inject as environment variables, and write logs to CloudWatch.

**Typical permissions**:
```json
{
  "Effect": "Allow",
  "Action": [
    "ecr:GetAuthorizationToken",
    "ecr:BatchCheckLayerAvailability",
    "ecr:GetDownloadUrlForLayer",
    "ecr:BatchGetImage",
    "logs:CreateLogStream",
    "logs:PutLogEvents",
    "secretsmanager:GetSecretValue",
    "ssm:GetParameters",
    "kms:Decrypt"
  ],
  "Resource": "*"
}
```

**Scoped at**: the ECS service level. If no `executionRoleArn` is specified, ECS uses `ecsTaskExecutionRole` (default managed role).

### Task Role (`taskRoleArn`)

**Who uses it**: Your application code running inside the container, via the container metadata credential endpoint (`169.254.170.2`).

**Purpose**: Grants your app permission to call AWS APIs at runtime — S3, DynamoDB, Bedrock, etc.

**Typical permissions for an AI API**:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "bedrock:InvokeModel",
    "sqs:SendMessage",
    "dynamodb:GetItem",
    "dynamodb:PutItem"
  ],
  "Resource": [
    "arn:aws:s3:::my-ai-data-bucket/*",
    "arn:aws:bedrock:us-east-1::foundation-model/*",
    "arn:aws:sqs:us-east-1:123456789012:ai-jobs",
    "arn:aws:dynamodb:us-east-1:123456789012:table/ai-sessions"
  ]
}
```

**Key distinction**: Neither role is hardcoded — the container gets short-lived credentials via the ECS metadata endpoint. This is the correct pattern; no static access keys in environment variables or code.

---

## Exercise 3: ECS Fargate vs EC2 Launch Type Tradeoffs

| Dimension | Fargate | EC2 launch type |
|-----------|---------|-----------------|
| **Infrastructure management** | AWS manages underlying hosts entirely | You provision, patch, and scale EC2 nodes |
| **GPU support** | Not supported | Supported (P3, P4, G4, G5 instance families) |
| **Pricing model** | Per vCPU-second + GB-second; no idle node cost | Per EC2 instance-hour; idle capacity still billed |
| **Cold start** | ~10–30 s (task + image pull) | Depends on node capacity; tasks start fast if node is warm |
| **Max CPU per task** | 16 vCPU, 120 GB RAM | Entire instance size (up to 192 vCPU on hpc7g.16xlarge) |
| **Bin packing efficiency** | AWS handles it; you can't co-locate tasks on the same host | You control placement strategies (binpack, spread) |
| **Spot support** | Fargate Spot: ~70% discount, can be interrupted | EC2 Spot: same discount; Spot Instance Interruption Notices give 2-minute warning |
| **Network isolation** | Each task gets ENI (awsvpc only) | Can use awsvpc or bridge network modes |
| **Compliance / dedicated hosts** | Not possible (shared Fargate fleet) | EC2 Dedicated Hosts supported; needed for BYOL licensing |
| **Operational complexity** | Low — no node autoscaling groups, no AMI updates | High — must manage Auto Scaling Group, ECS-optimized AMI, node draining |

**When to choose Fargate**: Teams prioritizing developer velocity, variable or unpredictable traffic, no GPU requirements. The cost premium is justified by eliminating ops overhead.

**When to choose EC2**: GPU inference, very high throughput needing bin-packing for cost, compliance requiring dedicated hosts, or workloads where the per-task CPU/RAM ceiling matters.

---

## Exercise 4: IAM Policy for ECS Task Calling Secrets Manager and S3

**Design principles applied**:
- Least privilege: scoped to specific secret ARNs and S3 bucket prefix.
- No wildcards on resource where ARN is known.
- KMS decrypt included — Secrets Manager secrets encrypted with customer-managed key need it.
- Read-only on secrets (no `PutSecretValue`).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadAppSecrets",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/ai-api/*"
      ]
    },
    {
      "Sid": "DecryptSecretsKMS",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": [
        "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123def456"
      ],
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"
        }
      }
    },
    {
      "Sid": "ReadS3ModelArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ai-models-bucket",
        "arn:aws:s3:::my-ai-models-bucket/models/*"
      ]
    },
    {
      "Sid": "WriteS3Outputs",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::my-ai-outputs-bucket/tasks/*"
      ]
    },
    {
      "Sid": "DenySecretsWrite",
      "Effect": "Deny",
      "Action": [
        "secretsmanager:PutSecretValue",
        "secretsmanager:DeleteSecret",
        "secretsmanager:RotateSecret"
      ],
      "Resource": "*"
    }
  ]
}
```

**Why the explicit Deny at the end**: Defense in depth. If an attacker compromises the container and gains the task role credentials, they cannot exfiltrate or rotate secrets. The Deny overrides any Allow even from broader policies.

**Trust policy for the task role** (attach this to the IAM role so ECS can assume it):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "ArnLike": {
          "aws:SourceArn": "arn:aws:ecs:us-east-1:123456789012:*"
        },
        "StringEquals": {
          "aws:SourceAccount": "123456789012"
        }
      }
    }
  ]
}
```

The `Condition` block prevents confused deputy attacks — only ECS tasks in your own account can assume this role.
