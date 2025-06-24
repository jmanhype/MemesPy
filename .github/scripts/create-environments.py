#!/usr/bin/env python3

import subprocess
import json
import sys

def run_gh_command(args):
    """Run a gh CLI command and return the output."""
    result = subprocess.run(['gh'] + args, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def create_environment(repo, env_name, protection_rules=None):
    """Create or update a GitHub environment."""
    print(f"Creating {env_name} environment...")
    
    # Base API call
    api_args = [
        'api', '-X', 'PUT',
        f'repos/{repo}/environments/{env_name}',
        '-H', 'Accept: application/vnd.github+json'
    ]
    
    # Add protection rules if specified
    if protection_rules:
        for key, value in protection_rules.items():
            if isinstance(value, (dict, list)):
                api_args.extend(['--field', f'{key}={json.dumps(value)}'])
            else:
                api_args.extend(['--field', f'{key}={value}'])
    
    stdout, stderr, returncode = run_gh_command(api_args)
    
    if returncode == 0:
        print(f"✅ {env_name} environment created/updated")
    else:
        print(f"❌ Failed to create {env_name} environment: {stderr}")

def create_secret(name, value, repo, environment=None):
    """Create a GitHub secret."""
    location = f"environment '{environment}'" if environment else "repository"
    print(f"Creating secret {name} in {location}...")
    
    # Use echo to pipe the value to gh secret set
    cmd = ['echo', '-n', value]
    gh_args = ['secret', 'set', name]
    
    if environment:
        gh_args.extend(['--env', environment])
    
    # Run echo and pipe to gh
    echo_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    gh_proc = subprocess.Popen(['gh'] + gh_args, stdin=echo_proc.stdout, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    echo_proc.stdout.close()
    stdout, stderr = gh_proc.communicate()
    
    if gh_proc.returncode == 0:
        print(f"✅ Secret {name} created")
    else:
        print(f"⚠️  Secret {name} might already exist")

def main():
    repo = "jmanhype/MemesPy"
    
    print("🔧 Setting up GitHub Environments and Secrets...")
    print()
    
    # Create environments
    print("📦 Creating environments...")
    
    # Staging environment (no protection)
    create_environment(repo, 'staging')
    
    # Production environment (with protection)
    # Note: Protection rules require proper permission scopes
    create_environment(repo, 'production', {
        'wait_timer': 5,  # 5 minute wait
        'deployment_branch_policy': {
            'protected_branches': False,
            'custom_branch_policies': False
        }
    })
    
    print()
    print("🔐 Creating secrets...")
    
    # Repository-wide secrets (placeholders - update with real values)
    secrets = {
        'OPENAI_API_KEY': 'sk-proj-PLACEHOLDER',
        'SONAR_TOKEN': 'sqp_PLACEHOLDER',
        'SLACK_WEBHOOK': 'https://hooks.slack.com/services/PLACEHOLDER'
    }
    
    for name, value in secrets.items():
        create_secret(name, value, repo)
    
    # Environment-specific secrets
    env_secrets = {
        'staging': {
            'DATABASE_URL': 'postgresql://memespy:password@localhost:5432/memespy_staging',
            'REDIS_URL': 'redis://localhost:6379/0',
            'SENTRY_DSN': 'https://PLACEHOLDER@sentry.io/staging',
            'ENVIRONMENT': 'staging'
        },
        'production': {
            'DATABASE_URL': 'postgresql://memespy:password@localhost:5432/memespy_prod',
            'REDIS_URL': 'redis://localhost:6379/1',
            'SENTRY_DSN': 'https://PLACEHOLDER@sentry.io/production',
            'ENVIRONMENT': 'production'
        }
    }
    
    for env, secrets in env_secrets.items():
        print(f"\nCreating {env} environment secrets...")
        for name, value in secrets.items():
            create_secret(name, value, repo, env)
    
    print()
    print("✅ Environment setup complete!")
    print()
    print("📋 Next steps:")
    print("1. Create project board at: https://github.com/jmanhype/MemesPy/projects/new")
    print("2. Update secret values in Settings → Secrets and variables")
    print("3. Test the pipeline by creating a PR")

if __name__ == "__main__":
    main()