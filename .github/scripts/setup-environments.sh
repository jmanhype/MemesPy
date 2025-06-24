#!/bin/bash

# Setup GitHub Environments and Secrets for MemesPy

echo "🔧 Setting up GitHub Environments and Secrets..."

# Create environments
echo "📦 Creating environments..."

# Create staging environment
gh api -X PUT repos/jmanhype/MemesPy/environments/staging \
  --field wait_timer=0 \
  --field reviewers='[]' \
  --field deployment_branch_policy='{"protected_branches":false,"custom_branch_policies":true}' \
  2>/dev/null || echo "Staging environment might already exist"

# Create production environment with protection rules
gh api -X PUT repos/jmanhype/MemesPy/environments/production \
  --field wait_timer=5 \
  --field reviewers='[{"type":"User","id":56135400}]' \
  --field deployment_branch_policy='{"protected_branches":false,"custom_branch_policies":true}' \
  2>/dev/null || echo "Production environment might already exist"

echo "✅ Environments created"

# Create repository secrets
echo "🔐 Creating repository secrets..."

# Function to create secret
create_secret() {
  local name=$1
  local value=$2
  local env=$3
  
  if [ -z "$env" ]; then
    # Repository secret
    echo -n "$value" | gh secret set "$name" 2>/dev/null || echo "Secret $name might already exist"
  else
    # Environment secret
    echo -n "$value" | gh secret set "$name" --env "$env" 2>/dev/null || echo "Secret $name in $env might already exist"
  fi
}

# Repository-wide secrets
echo "Creating repository secrets..."
create_secret "OPENAI_API_KEY" "your-openai-api-key-here"
create_secret "SONAR_TOKEN" "your-sonarqube-token-here"
create_secret "SLACK_WEBHOOK" "your-slack-webhook-here"

# Staging environment secrets
echo "Creating staging environment secrets..."
create_secret "DATABASE_URL" "postgresql://user:pass@staging-db:5432/memespy" "staging"
create_secret "REDIS_URL" "redis://staging-redis:6379" "staging"
create_secret "SENTRY_DSN" "https://staging@sentry.io/memespy" "staging"
create_secret "ENVIRONMENT" "staging" "staging"

# Production environment secrets
echo "Creating production environment secrets..."
create_secret "DATABASE_URL" "postgresql://user:pass@prod-db:5432/memespy" "production"
create_secret "REDIS_URL" "redis://prod-redis:6379" "production"
create_secret "SENTRY_DSN" "https://prod@sentry.io/memespy" "production"
create_secret "ENVIRONMENT" "production" "production"

echo "✅ Secrets created (with placeholder values)"

# Create project board
echo "📋 Creating project board..."
echo ""
echo "Please manually create the project board at:"
echo "https://github.com/jmanhype/MemesPy/projects/new"
echo ""
echo "Use these settings:"
echo "- Template: Board"
echo "- Name: MemesPy Development Pipeline"
echo "- Description: Production deployment pipeline following DevOps best practices"
echo ""
echo "Add these columns:"
echo "1. 📋 Backlog"
echo "2. 🎯 Ready" 
echo "3. 💻 In Development"
echo "4. 🔍 In Review"
echo "5. ✅ Done"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "⚠️  IMPORTANT: Update the placeholder secret values with real ones:"
echo "   - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys"
echo "   - DATABASE_URL: Your PostgreSQL connection string"
echo "   - REDIS_URL: Your Redis connection string"
echo "   - SONAR_TOKEN: From your SonarQube instance"
echo "   - SLACK_WEBHOOK: From your Slack app"
echo "   - SENTRY_DSN: From your Sentry project"