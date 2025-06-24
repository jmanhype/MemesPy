#!/bin/bash

# MemesPy Label Setup Script

echo "🏷️  Setting up GitHub labels for MemesPy..."

# Priority labels
gh label create "p0-critical" --color "FF0000" --description "System down, blocking all users" 2>/dev/null
gh label create "p1-high" --color "FF6B6B" --description "Major feature broken" 2>/dev/null
gh label create "p2-medium" --color "FFA500" --description "Minor feature issue" 2>/dev/null
gh label create "p3-low" --color "FFFF00" --description "Nice to have" 2>/dev/null

# Size labels
gh label create "size/S" --color "69D100" --description "< 4 hours" 2>/dev/null
gh label create "size/M" --color "29A88E" --description "1-2 days" 2>/dev/null
gh label create "size/L" --color "5319E7" --description "3-5 days" 2>/dev/null
gh label create "size/XL" --color "D93F0B" --description "> 1 week" 2>/dev/null

# Type labels
gh label create "feature" --color "A2EEEF" --description "New functionality" 2>/dev/null
gh label create "bug" --color "D73A4A" --description "Something broken" 2>/dev/null
gh label create "enhancement" --color "A8E6CF" --description "Improvement to existing feature" 2>/dev/null
gh label create "docs" --color "0075CA" --description "Documentation only" 2>/dev/null
gh label create "refactor" --color "D4C5F9" --description "Code improvement, no behavior change" 2>/dev/null

# Component labels
gh label create "api" --color "7057FF" --description "FastAPI endpoints" 2>/dev/null
gh label create "agents" --color "008672" --description "DSPy agents" 2>/dev/null
gh label create "database" --color "F9D71C" --description "Models and migrations" 2>/dev/null
gh label create "frontend" --color "B4E7CE" --description "UI components" 2>/dev/null
gh label create "ci-cd" --color "0E8A16" --description "Pipeline and deployment" 2>/dev/null

# Status labels
gh label create "needs-triage" --color "FBCA04" --description "Needs review" 2>/dev/null
gh label create "ready" --color "0E8A16" --description "Definition of Ready met" 2>/dev/null
gh label create "blocked" --color "E99695" --description "Waiting on dependency" 2>/dev/null
gh label create "in-progress" --color "1D76DB" --description "Being worked on" 2>/dev/null

echo "✅ Labels created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://github.com/jmanhype/MemesPy/projects/new"
echo "2. Create a project board named 'MemesPy Development Pipeline'"
echo "3. Add columns: Backlog, Ready, In Development, In Review, Done"
echo "4. Link the project to your repository"