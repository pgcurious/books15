# Module 7.2: Cloud Deployment

> "The cloud is not just someone else's computer—it's someone else's operations team." — Anonymous DevOps Engineer

## What You'll Learn

- How to choose the right cloud platform and service for your agent
- Deploy containerized agents to AWS, GCP, and Azure
- Set up CI/CD pipelines for automated deployments
- Manage secrets and configuration across environments
- Implement blue-green and canary deployments
- Monitor and troubleshoot cloud deployments

---

## First Principles: What Is Cloud Deployment?

### Breaking Down the Problem

At its core, cloud deployment answers one question: **How do we run our container where users can access it?**

```
CLOUD DEPLOYMENT = COMPUTE + NETWORKING + STORAGE + MANAGEMENT

Where:
├── COMPUTE: Running your container
│   ├── Where does the code execute?
│   ├── How much CPU/memory?
│   └── How many instances?
│
├── NETWORKING: Exposing your container
│   ├── How do users reach it?
│   ├── How is traffic distributed?
│   └── How is it secured?
│
├── STORAGE: Persistent data
│   ├── Environment variables
│   ├── Secrets
│   └── Logs and metrics
│
└── MANAGEMENT: Operations
    ├── How do we deploy updates?
    ├── How do we scale?
    └── How do we monitor?
```

### The Cloud Service Spectrum

```
ABSTRACTION LEVEL
─────────────────────────────────────────────────────────────────

High Abstraction                              Low Abstraction
(Less Control, Less Work)                     (More Control, More Work)

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   SERVERLESS │  │  CONTAINER   │  │  KUBERNETES  │  │     VMs      │
│              │  │  SERVICES    │  │              │  │              │
│  - Lambda    │  │  - Cloud Run │  │  - EKS       │  │  - EC2       │
│  - Functions │  │  - App Runner│  │  - GKE       │  │  - GCE       │
│  - Azure Func│  │  - Container │  │  - AKS       │  │  - Azure VMs │
│              │  │    Apps      │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
      │                 │                 │                 │
      │                 │                 │                 │
      ▼                 ▼                 ▼                 ▼
   You manage:      You manage:       You manage:      You manage:
   - Code only      - Containers      - Pods/Services  - Everything
                    - Basic config    - Ingress        - OS patches
                                      - Networking     - Security
                                                      - Networking
```

**For AI Agents**: Container services (Cloud Run, App Runner, Container Apps) hit the sweet spot—enough control for production needs, without Kubernetes complexity.

---

## Analogical Thinking: Cloud Deployment as Real Estate

```
REAL ESTATE                           CLOUD DEPLOYMENT
───────────────────────────────────────────────────────────────────

Hotel Room (Serverless)               Lambda/Cloud Functions
├── Pay per night                     ├── Pay per request
├── Cleaning included                 ├── Scaling included
├── Limited customization             ├── Limited runtime
└── Great for short stays             └── Great for sporadic traffic

Apartment (Container Services)        Cloud Run/App Runner
├── Monthly rent                      ├── Pay for usage
├── Some customization                ├── Configure resources
├── Building maintenance handled      ├── Platform handles infra
└── Good for most people              └── Good for most workloads

House (Kubernetes)                    EKS/GKE/AKS
├── Mortgage/ownership                ├── Cluster management
├── Full customization                ├── Full control
├── You handle maintenance            ├── You handle operations
└── For specific needs                └── For complex requirements

Build Custom (VMs)                    EC2/GCE/Azure VMs
├── Design everything                 ├── Configure everything
├── Total control                     ├── Total responsibility
├── Maximum flexibility               ├── Maximum complexity
└── For unique requirements           └── For specialized needs
```

**Key insight**: Most AI agents belong in the "apartment" category—container services that handle infrastructure while giving you enough control.

---

## AWS Deployment

### Option 1: AWS App Runner (Simplest)

App Runner is AWS's fully managed container service—perfect for getting started.

```
AWS APP RUNNER ARCHITECTURE
─────────────────────────────────────────────────────────────────

                         Internet
                             │
                             ▼
                    ┌─────────────────┐
                    │  App Runner     │
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │Container │  │Container │  │Container │
        │ Instance │  │ Instance │  │ Instance │
        │    1     │  │    2     │  │    n     │
        └──────────┘  └──────────┘  └──────────┘
                             │
                    (Auto-scaling based on traffic)
```

**Deployment Steps:**

```bash
# 1. Install AWS CLI and configure
aws configure

# 2. Create ECR repository
aws ecr create-repository --repository-name ai-agent

# 3. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build -t ai-agent:latest .
docker tag ai-agent:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-agent:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-agent:latest

# 4. Create App Runner service
aws apprunner create-service \
  --service-name ai-agent \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$AWS_ACCOUNT_ID'.dkr.ecr.us-east-1.amazonaws.com/ai-agent:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {
          "OPENAI_API_KEY": "{{resolve:secretsmanager:openai-key}}"
        }
      }
    },
    "AutoDeploymentsEnabled": true,
    "AuthenticationConfiguration": {
      "AccessRoleArn": "arn:aws:iam::'$AWS_ACCOUNT_ID':role/AppRunnerECRAccessRole"
    }
  }' \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \
  --auto-scaling-configuration-arn "arn:aws:apprunner:us-east-1:$AWS_ACCOUNT_ID:autoscalingconfiguration/DefaultConfiguration/1"
```

**Infrastructure as Code (Terraform):**

```hcl
# terraform/aws/apprunner.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECR Repository
resource "aws_ecr_repository" "agent" {
  name                 = "ai-agent"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# IAM Role for App Runner
resource "aws_iam_role" "apprunner_ecr" {
  name = "apprunner-ecr-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "build.apprunner.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# Secrets Manager for API Key
resource "aws_secretsmanager_secret" "openai_key" {
  name = "openai-api-key"
}

# App Runner Service
resource "aws_apprunner_service" "agent" {
  service_name = "ai-agent"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.agent.repository_url}:latest"
      image_repository_type = "ECR"

      image_configuration {
        port = "8000"
        runtime_environment_secrets = {
          OPENAI_API_KEY = aws_secretsmanager_secret.openai_key.arn
        }
      }
    }

    auto_deployments_enabled = true
  }

  instance_configuration {
    cpu    = "1024"  # 1 vCPU
    memory = "2048"  # 2 GB
  }

  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.agent.arn

  health_check_configuration {
    protocol            = "HTTP"
    path               = "/health"
    interval           = 10
    timeout            = 5
    healthy_threshold  = 1
    unhealthy_threshold = 5
  }

  tags = {
    Environment = var.environment
    Project     = "ai-agent"
  }
}

resource "aws_apprunner_auto_scaling_configuration_version" "agent" {
  auto_scaling_configuration_name = "ai-agent-scaling"
  max_concurrency                 = 100
  max_size                        = 10
  min_size                        = 1
}

output "service_url" {
  value = aws_apprunner_service.agent.service_url
}
```

### Option 2: AWS ECS with Fargate

For more control and features:

```hcl
# terraform/aws/ecs.tf

# VPC and Networking
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "ai-agent-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "ai-agent-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "agent" {
  family                   = "ai-agent"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "agent"
    image = "${aws_ecr_repository.agent.repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "LOG_LEVEL", value = "INFO" }
    ]

    secrets = [{
      name      = "OPENAI_API_KEY"
      valueFrom = aws_secretsmanager_secret.openai_key.arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/ai-agent"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "agent"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

# ECS Service
resource "aws_ecs_service" "agent" {
  name            = "ai-agent"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.agent.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent.arn
    container_name   = "agent"
    container_port   = 8000
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  depends_on = [aws_lb_listener.agent]
}

# Application Load Balancer
resource "aws_lb" "agent" {
  name               = "ai-agent-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
}

resource "aws_lb_target_group" "agent" {
  name        = "ai-agent-tg"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = module.vpc.vpc_id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 5
    interval            = 30
  }
}

resource "aws_lb_listener" "agent" {
  load_balancer_arn = aws_lb.agent.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.agent.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agent.arn
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "agent" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.agent.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-auto-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.agent.resource_id
  scalable_dimension = aws_appautoscaling_target.agent.scalable_dimension
  service_namespace  = aws_appautoscaling_target.agent.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

---

## GCP Deployment

### Option 1: Cloud Run (Recommended)

Cloud Run is GCP's serverless container platform—excellent for AI agents.

```
GCP CLOUD RUN ARCHITECTURE
─────────────────────────────────────────────────────────────────

                         Internet
                             │
                             ▼
                    ┌─────────────────┐
                    │  Cloud Run      │
                    │  Ingress        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Instance │  │ Instance │  │ Instance │
        │    1     │  │    2     │  │    n     │
        └──────────┘  └──────────┘  └──────────┘
                             │
                    (Scales to zero when idle)
```

**Deployment Steps:**

```bash
# 1. Configure gcloud
gcloud auth login
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# 3. Create Artifact Registry repository
gcloud artifacts repositories create ai-agents \
  --repository-format=docker \
  --location=us-central1

# 4. Build and push with Cloud Build
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/ai-agents/agent:latest

# 5. Create secret for API key
echo -n $OPENAI_API_KEY | gcloud secrets create openai-api-key --data-file=-

# 6. Deploy to Cloud Run
gcloud run deploy ai-agent \
  --image us-central1-docker.pkg.dev/$PROJECT_ID/ai-agents/agent:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 1 \
  --memory 2Gi \
  --min-instances 1 \
  --max-instances 10 \
  --concurrency 80 \
  --timeout 300 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

**Infrastructure as Code (Terraform):**

```hcl
# terraform/gcp/cloudrun.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Artifact Registry
resource "google_artifact_registry_repository" "agents" {
  location      = var.region
  repository_id = "ai-agents"
  format        = "DOCKER"
}

# Secret Manager
resource "google_secret_manager_secret" "openai_key" {
  secret_id = "openai-api-key"

  replication {
    auto {}
  }
}

# Service Account
resource "google_service_account" "agent" {
  account_id   = "ai-agent-sa"
  display_name = "AI Agent Service Account"
}

# Grant secret access
resource "google_secret_manager_secret_iam_member" "agent_secret_access" {
  secret_id = google_secret_manager_secret.openai_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "agent" {
  name     = "ai-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent.email

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/ai-agents/agent:latest"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle = true  # Scale to zero when idle
      }

      env {
        name = "LOG_LEVEL"
        value = "INFO"
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds       = 10
        failure_threshold    = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        timeout_seconds   = 5
        period_seconds   = 30
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow public access
resource "google_cloud_run_service_iam_member" "public" {
  location = google_cloud_run_v2_service.agent.location
  service  = google_cloud_run_v2_service.agent.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "service_url" {
  value = google_cloud_run_v2_service.agent.uri
}
```

### Traffic Splitting for Canary Deployments

```hcl
# Canary deployment with Cloud Run
resource "google_cloud_run_v2_service" "agent" {
  # ... previous config ...

  traffic {
    type     = "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION"
    revision = google_cloud_run_v2_service.agent.latest_created_revision
    percent  = 90
    tag      = "stable"
  }

  traffic {
    type     = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent  = 10
    tag      = "canary"
  }
}
```

---

## Azure Deployment

### Option 1: Azure Container Apps

```
AZURE CONTAINER APPS ARCHITECTURE
─────────────────────────────────────────────────────────────────

                         Internet
                             │
                             ▼
                    ┌─────────────────┐
                    │  Container Apps │
                    │  Environment    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Replica  │  │ Replica  │  │ Replica  │
        │    1     │  │    2     │  │    n     │
        └──────────┘  └──────────┘  └──────────┘
                             │
                    (KEDA-based auto-scaling)
```

**Deployment Steps:**

```bash
# 1. Install Azure CLI and login
az login
az account set --subscription $SUBSCRIPTION_ID

# 2. Create resource group
az group create --name ai-agent-rg --location eastus

# 3. Create Container Registry
az acr create --resource-group ai-agent-rg --name aiagentacr --sku Basic
az acr login --name aiagentacr

# 4. Build and push image
docker tag ai-agent:latest aiagentacr.azurecr.io/ai-agent:latest
docker push aiagentacr.azurecr.io/ai-agent:latest

# 5. Create Container Apps environment
az containerapp env create \
  --name ai-agent-env \
  --resource-group ai-agent-rg \
  --location eastus

# 6. Create Key Vault and secret
az keyvault create --name ai-agent-kv --resource-group ai-agent-rg --location eastus
az keyvault secret set --vault-name ai-agent-kv --name openai-api-key --value $OPENAI_API_KEY

# 7. Deploy Container App
az containerapp create \
  --name ai-agent \
  --resource-group ai-agent-rg \
  --environment ai-agent-env \
  --image aiagentacr.azurecr.io/ai-agent:latest \
  --registry-server aiagentacr.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --cpu 1 \
  --memory 2Gi \
  --min-replicas 1 \
  --max-replicas 10 \
  --secrets openai-key=keyvaultref:ai-agent-kv/openai-api-key \
  --env-vars OPENAI_API_KEY=secretref:openai-key
```

**Infrastructure as Code (Terraform):**

```hcl
# terraform/azure/containerapp.tf

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "main" {
  name     = "ai-agent-rg"
  location = var.location
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "aiagentacr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location           = azurerm_resource_group.main.location
  sku                = "Basic"
  admin_enabled      = true
}

# Log Analytics for monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "ai-agent-logs"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                = "PerGB2018"
  retention_in_days  = 30
}

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "ai-agent-env"
  location                  = azurerm_resource_group.main.location
  resource_group_name       = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
}

# Key Vault for secrets
resource "azurerm_key_vault" "main" {
  name                = "ai-agent-kv-${random_string.suffix.result}"
  location           = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id          = data.azurerm_client_config.current.tenant_id
  sku_name           = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = ["Get", "List", "Set", "Delete"]
  }
}

resource "azurerm_key_vault_secret" "openai_key" {
  name         = "openai-api-key"
  value        = var.openai_api_key
  key_vault_id = azurerm_key_vault.main.id
}

# Container App
resource "azurerm_container_app" "agent" {
  name                         = "ai-agent"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name         = azurerm_resource_group.main.name
  revision_mode               = "Single"

  secret {
    name  = "openai-api-key"
    value = var.openai_api_key
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username            = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  template {
    container {
      name   = "agent"
      image  = "${azurerm_container_registry.main.login_server}/ai-agent:latest"
      cpu    = 1.0
      memory = "2Gi"

      env {
        name        = "OPENAI_API_KEY"
        secret_name = "openai-api-key"
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      liveness_probe {
        transport = "HTTP"
        path      = "/health"
        port      = 8000
      }

      readiness_probe {
        transport = "HTTP"
        path      = "/ready"
        port      = 8000
      }
    }

    min_replicas = 1
    max_replicas = 10
  }

  ingress {
    external_enabled = true
    target_port     = 8000
    traffic_weight {
      percentage = 100
      latest_revision = true
    }
  }
}

output "app_url" {
  value = "https://${azurerm_container_app.agent.ingress[0].fqdn}"
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}
```

---

## CI/CD Pipelines

### GitHub Actions (Works with All Clouds)

```yaml
# .github/workflows/deploy.yml

name: Deploy AI Agent

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ai-agent

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio httpx

      - name: Run tests
        run: pytest tests/ -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-aws:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Push to ECR
        run: |
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker tag ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            ${{ steps.login-ecr.outputs.registry }}/ai-agent:${{ github.sha }}
          docker push ${{ steps.login-ecr.outputs.registry }}/ai-agent:${{ github.sha }}

      - name: Deploy to App Runner
        run: |
          aws apprunner update-service \
            --service-arn ${{ secrets.APPRUNNER_SERVICE_ARN }} \
            --source-configuration '{
              "ImageRepository": {
                "ImageIdentifier": "${{ steps.login-ecr.outputs.registry }}/ai-agent:${{ github.sha }}",
                "ImageRepositoryType": "ECR"
              }
            }'

  deploy-gcp:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker
        run: gcloud auth configure-docker us-central1-docker.pkg.dev

      - name: Push to Artifact Registry
        run: |
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker tag ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ai-agents/agent:${{ github.sha }}
          docker push us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ai-agents/agent:${{ github.sha }}

      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: ai-agent
          region: us-central1
          image: us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT }}/ai-agents/agent:${{ github.sha }}
```

---

## Secrets Management

### Best Practices

```
SECRETS MANAGEMENT HIERARCHY
─────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│                    DO NOT DO THIS                                │
├─────────────────────────────────────────────────────────────────┤
│  ✗ Hardcode secrets in code                                      │
│  ✗ Commit .env files to git                                     │
│  ✗ Pass secrets as command-line arguments                       │
│  ✗ Store secrets in container images                            │
│  ✗ Log secrets anywhere                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DO THIS INSTEAD                               │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Use cloud secret managers:                                    │
│    - AWS Secrets Manager                                         │
│    - GCP Secret Manager                                          │
│    - Azure Key Vault                                             │
│                                                                  │
│  ✓ Inject secrets at runtime via environment variables          │
│  ✓ Rotate secrets regularly                                      │
│  ✓ Use different secrets for different environments             │
│  ✓ Audit secret access                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Accessing Secrets in Code

```python
# app/config.py - Secure configuration handling

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with secret injection."""

    # Application
    app_name: str = "ai-agent"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # API Keys (injected from secrets manager)
    openai_api_key: str
    langchain_api_key: Optional[str] = None

    # Optional cloud-specific
    aws_region: str = "us-east-1"
    gcp_project: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_cloud_secrets(cls) -> "Settings":
        """Load secrets from cloud provider."""
        environment = os.getenv("ENVIRONMENT", "development")

        if environment == "production":
            # In production, secrets come from environment variables
            # which are injected by the cloud provider
            return cls()
        else:
            # In development, load from .env file
            return cls(_env_file=".env")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_cloud_secrets()
```

---

## Deployment Patterns

### Blue-Green Deployment

```
BLUE-GREEN DEPLOYMENT
─────────────────────────────────────────────────────────────────

Before deployment:
┌──────────────────────────────────────────────────────────────┐
│                         Load Balancer                         │
│                              │                                │
│                        100% traffic                          │
│                              │                                │
│                              ▼                                │
│                     ┌───────────────┐                        │
│                     │   BLUE (v1)   │ ← Active               │
│                     │   3 instances │                        │
│                     └───────────────┘                        │
│                                                               │
│                     ┌───────────────┐                        │
│                     │  GREEN (v2)   │ ← Idle (deploying)     │
│                     │   3 instances │                        │
│                     └───────────────┘                        │
└──────────────────────────────────────────────────────────────┘

After deployment (instant switch):
┌──────────────────────────────────────────────────────────────┐
│                         Load Balancer                         │
│                              │                                │
│                        100% traffic                          │
│                              │                                │
│                              ▼                                │
│                     ┌───────────────┐                        │
│                     │  GREEN (v2)   │ ← Active               │
│                     │   3 instances │                        │
│                     └───────────────┘                        │
│                                                               │
│                     ┌───────────────┐                        │
│                     │   BLUE (v1)   │ ← Standby (rollback)   │
│                     │   3 instances │                        │
│                     └───────────────┘                        │
└──────────────────────────────────────────────────────────────┘
```

### Canary Deployment

```
CANARY DEPLOYMENT
─────────────────────────────────────────────────────────────────

Phase 1: Initial canary (5%)
┌──────────────────────────────────────────────────────────────┐
│                         Load Balancer                         │
│                        ┌─────┴─────┐                         │
│                        │           │                          │
│                     95%│           │5%                        │
│                        ▼           ▼                          │
│               ┌─────────────┐ ┌─────────────┐                │
│               │  STABLE     │ │  CANARY     │                │
│               │  (v1)       │ │  (v2)       │                │
│               │  10 inst.   │ │  1 inst.    │                │
│               └─────────────┘ └─────────────┘                │
│                                                               │
│  Monitor: Error rate, latency, user feedback                 │
└──────────────────────────────────────────────────────────────┘

Phase 2: Gradual rollout (25%)
┌──────────────────────────────────────────────────────────────┐
│                        75%          25%                       │
│                         ▼           ▼                          │
│               ┌─────────────┐ ┌─────────────┐                │
│               │  STABLE     │ │  CANARY     │                │
│               │  (v1)       │ │  (v2)       │                │
│               │  8 inst.    │ │  3 inst.    │                │
│               └─────────────┘ └─────────────┘                │
└──────────────────────────────────────────────────────────────┘

Phase 3: Full rollout (100%)
┌──────────────────────────────────────────────────────────────┐
│                        100% traffic                           │
│                              ▼                                │
│                     ┌─────────────┐                          │
│                     │  NEW (v2)   │                          │
│                     │  10 inst.   │                          │
│                     └─────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Monitoring and Troubleshooting

### Essential Metrics to Track

```python
# app/monitoring.py

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Request metrics
REQUEST_COUNT = Counter(
    'agent_requests_total',
    'Total requests to the agent',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'agent_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# LLM metrics
LLM_REQUESTS = Counter(
    'agent_llm_requests_total',
    'Total LLM API calls',
    ['model', 'status']
)

LLM_TOKENS = Counter(
    'agent_llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt, completion
)

LLM_LATENCY = Histogram(
    'agent_llm_latency_seconds',
    'LLM response latency',
    ['model']
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    'agent_active_requests',
    'Currently processing requests'
)

CACHE_HITS = Counter(
    'agent_cache_hits_total',
    'Cache hit count'
)

CACHE_MISSES = Counter(
    'agent_cache_misses_total',
    'Cache miss count'
)
```

### Cloud-Specific Monitoring

```bash
# AWS CloudWatch Logs Insights query
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100

# GCP Cloud Logging query
resource.type="cloud_run_revision"
resource.labels.service_name="ai-agent"
severity>=ERROR

# Azure Log Analytics query
ContainerAppConsoleLogs_CL
| where ContainerName_s == "ai-agent"
| where Log_s contains "ERROR"
| sort by TimeGenerated desc
```

---

## Key Takeaways

### 1. Choose the Right Abstraction Level
Start with managed container services (Cloud Run, App Runner) before moving to Kubernetes.

### 2. Automate Everything
CI/CD pipelines ensure consistent, repeatable deployments. Never deploy manually to production.

### 3. Use Native Secret Management
Each cloud has a secret manager—use it. Never put secrets in code or config files.

### 4. Implement Health Checks
Health and readiness endpoints enable orchestrators to manage your service lifecycle.

### 5. Deploy Gradually
Use canary or blue-green deployments to catch issues before they affect all users.

### 6. Monitor What Matters
Track request latency, error rates, and LLM token usage. Set up alerts for anomalies.

---

## What's Next?

In **Module 7.3: Cost Optimization & Scaling**, we'll learn:
- How to reduce LLM costs by 50-80%
- Smart caching strategies
- Auto-scaling policies that balance cost and performance
- MCP and A2A protocols for enterprise agent communication

You've deployed your agent—now let's make it economically sustainable!

[Continue to Module 7.3 →](03_cost_optimization_scaling.md)
