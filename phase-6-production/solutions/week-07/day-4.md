# Solutions — Week 07 · Day 4

## Exercise 1 — variables.tf with validation and sensitive

```hcl
# variables.tf

variable "instance_count" {
  type        = number
  default     = 1
  description = "Number of service instances to run"

  validation {
    condition     = var.instance_count >= 1 && var.instance_count <= 10
    error_message = "instance_count must be between 1 and 10 (inclusive)"
  }
}

variable "api_key" {
  type        = string
  sensitive   = true
  description = "API key for the external service"
  # Pass via: TF_VAR_api_key env var or .tfvars file (must be gitignored)
}
```

---

## Exercise 2 — terraform taint vs. terraform import

**`terraform taint` (deprecated in Terraform >= 0.15.2; replaced by `-replace` flag)**

Marks a resource for forced replacement on the next `apply`, even if no configuration change would trigger it. Use case: a resource is in a broken state that Terraform cannot detect (e.g., an EC2 instance that appears running but has a corrupted OS disk).

```bash
# Modern equivalent
terraform apply -replace="aws_instance.web"
```

**`terraform import`**

Brings an existing real-world resource under Terraform management — it adds the resource to the state file without changing the resource itself. Use case: infrastructure was created manually (click-ops) and you want to start managing it with Terraform without destroying and recreating it.

```bash
terraform import aws_s3_bucket.artifacts my-existing-bucket-name
```

After import, you must write the matching HCL resource block manually; Terraform does not generate it automatically (though `terraform plan -generate-config-out` can scaffold it in Terraform >= 1.5).

---

## Exercise 3 — State corruption without DynamoDB locking

**What happens without a lock table:**

1. Engineer A runs `terraform apply` — reads state (version 42), computes plan, starts creating resources.
2. Engineer B concurrently runs `terraform apply` — also reads state (version 42), computes a different plan.
3. Engineer A finishes, writes state (version 43) — new resources recorded.
4. Engineer B finishes, writes state (version 43, overwriting A's version) — Engineer A's newly created resources are no longer in state.

Result: "orphaned" resources — real cloud resources exist but are invisible to Terraform. Future `apply` runs may try to create duplicates, failing with "resource already exists" errors. `terraform destroy` will miss the orphaned resources (infra cost leakage).

**Recovery from state corruption:**

1. Run `terraform refresh` to sync actual cloud state into the state file.
2. Use `terraform state list` and `terraform state show` to audit what's in state vs. what exists in the cloud.
3. For orphaned resources, use `terraform import` to add them back to state.
4. For duplicated resources, use `terraform state rm` to remove the unwanted entry, then `terraform destroy -target=<resource>` to clean up the actual cloud resource.

**Prevention:** Always use DynamoDB locking. Recreate the lock table immediately if it is deleted.

---

## Exercise 4 — Dynamic AMI lookup with data block

```hcl
# Look up the latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Reference in a resource:
resource "aws_instance" "app" {
  ami           = data.aws_ami.amazon_linux_2023.id
  instance_type = "t3.micro"
}
```

**Why dynamic lookup is important:**

- Security patches are released as new AMI versions. Hardcoding `ami-0abcdef1234567890` means you never get security updates without a manual Terraform change.
- AMI IDs are region-specific. A hardcoded ID from `eu-west-1` fails when you deploy the same config in `us-east-1`. A dynamic lookup with `owners = ["amazon"]` works in any region.
- Dynamic lookups make infrastructure portable and self-healing against AWS deprecating old AMI versions.

---

## Exercise 5 — prevent_destroy with a force-replace plan

If a resource has `lifecycle { prevent_destroy = true }` and the plan shows `-/+` (replace), running `terraform apply` will fail with:

```
Error: Instance cannot be destroyed
  Resource aws_db_instance.main has lifecycle.prevent_destroy set,
  but the plan calls for this resource to be destroyed.
```

Terraform refuses the apply entirely — it does not perform any partial changes.

**How to override for a legitimate replacement:**

Option 1 (safest): Temporarily remove `prevent_destroy`, apply, re-add it.
```hcl
lifecycle {
  prevent_destroy = false   # temporary — re-add after apply
}
```

Option 2: Use `terraform apply -target=aws_db_instance.main` after removing `prevent_destroy` for a targeted replacement, minimizing the blast radius.

Option 3 (for data-safe replacements): Use `create_before_destroy = true` alongside the replacement to ensure the new resource is healthy before the old one is destroyed — and then re-enable `prevent_destroy` after the migration.

Never use `terraform destroy` on a database with `prevent_destroy` enabled — this is the protection working as intended. Always take a manual snapshot before a planned RDS replacement.
