# Azure MLOps Infrastructure
terraform {
  required_version = ">= 1.5"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Remote state backend (recommended for production)
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tfstate"
  #   container_name       = "tfstate"
  #   key                  = "mlops.tfstate"
  # }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Variables
variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "asset-mlops"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "francecentral"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Random suffix for globally unique names
resource "random_integer" "suffix" {
  min = 1000
  max = 9999
}

locals {
  resource_suffix = "${var.environment}-${random_integer.suffix.result}"
  common_tags = {
    Project     = "Asset Management MLOps"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Owner       = "DataTeam"
  }
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project_name}-${local.resource_suffix}"
  location = var.location
  tags     = local.common_tags
}

# Storage Account
resource "azurerm_storage_account" "main" {
  name                     = "st${replace(var.project_name, "-", "")}${random_integer.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
  }
  
  tags = local.common_tags
}

# Storage Containers
resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "reports" {
  name                  = "reports"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# App Service Plan
resource "azurerm_service_plan" "main" {
  name                = "asp-${var.project_name}-${local.resource_suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "B1"
  
  tags = local.common_tags
}

# Web App (Dashboard/API)
resource "azurerm_linux_web_app" "main" {
  name                = "app-${var.project_name}-${local.resource_suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  site_config {
    application_stack {
      python_version = "3.11"
    }
    
    always_on = false
  }

  app_settings = {
    AZURE_STORAGE_ACCOUNT_NAME   = azurerm_storage_account.main.name
    AZURE_STORAGE_CONNECTION_STR = azurerm_storage_account.main.primary_connection_string
    MLFLOW_TRACKING_URI          = "file:./mlruns"
    ENVIRONMENT                  = var.environment
  }
  
  https_only = true
  tags       = local.common_tags
}

# Outputs
output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Storage account name"
  value       = azurerm_storage_account.main.name
}

output "storage_connection_string" {
  description = "Storage connection string"
  value       = azurerm_storage_account.main.primary_connection_string
  sensitive   = true
}

output "web_app_url" {
  description = "Web app URL"
  value       = "https://${azurerm_linux_web_app.main.default_hostname}"
}

output "web_app_name" {
  description = "Web app name"
  value       = azurerm_linux_web_app.main.name
}
