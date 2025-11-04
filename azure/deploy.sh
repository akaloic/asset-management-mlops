#!/bin/bash
# Azure MLOps Asset Management Deployment Script

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="asset-mlops"
LOCATION="${AZURE_LOCATION:-francecentral}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
RANDOM_SUFFIX=$(date +%s | tail -c 5)

# Resource names
RESOURCE_GROUP="rg-${PROJECT_NAME}-${ENVIRONMENT}"
STORAGE_ACCOUNT="st${PROJECT_NAME}${RANDOM_SUFFIX}"
APP_SERVICE_PLAN="asp-${PROJECT_NAME}-${ENVIRONMENT}"
WEB_APP_NAME="app-${PROJECT_NAME}-${ENVIRONMENT}-${RANDOM_SUFFIX}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not found. Install from: https://aka.ms/azure-cli"
        exit 1
    fi
    
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Run: az login"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create resource group
create_resource_group() {
    log_info "Creating resource group: ${RESOURCE_GROUP}..."
    
    az group create \
        --name "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --tags Project="${PROJECT_NAME}" Environment="${ENVIRONMENT}" ManagedBy="Script" \
        --output none
    
    log_info "Resource group created successfully"
}

# Create storage account
create_storage_account() {
    log_info "Creating storage account: ${STORAGE_ACCOUNT}..."
    
    az storage account create \
        --name "${STORAGE_ACCOUNT}" \
        --resource-group "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --sku Standard_LRS \
        --https-only true \
        --min-tls-version TLS1_2 \
        --allow-blob-public-access false \
        --tags Project="${PROJECT_NAME}" Environment="${ENVIRONMENT}" \
        --output none
    
    log_info "Storage account created successfully"
}

# Create storage containers
create_storage_containers() {
    log_info "Creating storage containers..."
    
    for container in "data" "models" "reports"; do
        az storage container create \
            --name "${container}" \
            --account-name "${STORAGE_ACCOUNT}" \
            --auth-mode login \
            --output none || log_warn "Container ${container} may already exist"
    done
    
    log_info "Storage containers created successfully"
}

# Create app service plan
create_app_service_plan() {
    log_info "Creating App Service Plan: ${APP_SERVICE_PLAN}..."
    
    az appservice plan create \
        --name "${APP_SERVICE_PLAN}" \
        --resource-group "${RESOURCE_GROUP}" \
        --location "${LOCATION}" \
        --is-linux \
        --sku B1 \
        --tags Project="${PROJECT_NAME}" Environment="${ENVIRONMENT}" \
        --output none
    
    log_info "App Service Plan created successfully"
}

# Create web app
create_web_app() {
    log_info "Creating Web App: ${WEB_APP_NAME}..."
    
    az webapp create \
        --name "${WEB_APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --plan "${APP_SERVICE_PLAN}" \
        --runtime "PYTHON:3.11" \
        --tags Project="${PROJECT_NAME}" Environment="${ENVIRONMENT}" \
        --output none
    
    # Get storage connection string
    local conn_str=$(az storage account show-connection-string \
        --name "${STORAGE_ACCOUNT}" \
        --resource-group "${RESOURCE_GROUP}" \
        --output tsv)
    
    # Configure app settings
    az webapp config appsettings set \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${WEB_APP_NAME}" \
        --settings \
            AZURE_STORAGE_ACCOUNT_NAME="${STORAGE_ACCOUNT}" \
            AZURE_STORAGE_CONNECTION_STR="${conn_str}" \
            MLFLOW_TRACKING_URI="file:./mlruns" \
            ENVIRONMENT="${ENVIRONMENT}" \
        --output none
    
    log_info "Web App created and configured successfully"
}

# Main deployment
main() {
    echo "=========================================="
    echo "Azure MLOps Deployment"
    echo "=========================================="
    echo "Project: ${PROJECT_NAME}"
    echo "Environment: ${ENVIRONMENT}"
    echo "Location: ${LOCATION}"
    echo "=========================================="
    
    check_prerequisites
    create_resource_group
    create_storage_account
    create_storage_containers
    create_app_service_plan
    create_web_app
    
    echo ""
    echo "=========================================="
    log_info "Deployment completed successfully!"
    echo "=========================================="
    echo "Resources:"
    echo "  Resource Group: ${RESOURCE_GROUP}"
    echo "  Storage Account: ${STORAGE_ACCOUNT}"
    echo "  Web App: ${WEB_APP_NAME}"
    echo "  URL: https://${WEB_APP_NAME}.azurewebsites.net"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Deploy code: az webapp up --name ${WEB_APP_NAME} --resource-group ${RESOURCE_GROUP}"
    echo "  2. View logs: az webapp log tail --name ${WEB_APP_NAME} --resource-group ${RESOURCE_GROUP}"
    echo "  3. Open portal: az webapp browse --name ${WEB_APP_NAME} --resource-group ${RESOURCE_GROUP}"
}

# Run main with error handling
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap 'log_error "Deployment failed at line $LINENO"' ERR
    main "$@"
fi
