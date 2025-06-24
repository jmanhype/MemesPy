# MemesPy Development Pipeline Setup

## 🚀 Quick Setup

### 1. Create GitHub Project
1. Go to https://github.com/jmanhype/MemesPy/projects
2. Click "New project" → "Board" template
3. Name: "MemesPy Development Pipeline"

### 2. Configure Columns
Create these columns in order:
- **📋 Backlog** - New issues and ideas
- **🎯 Ready** - Meets Definition of Ready
- **💻 In Development** - Actively being worked on
- **🔍 In Review** - PR submitted
- **✅ Done** - Merged and deployed

### 3. Automation Rules

#### Backlog → Ready
When issue has all checkboxes in "Definition of Ready" checked

#### Ready → In Development  
When issue is assigned and branch created

#### In Development → In Review
When PR is opened linking to issue

#### In Review → Done
When PR is merged

## 📊 Issue Labels

### Priority
- `p0-critical` - System down, blocking all users
- `p1-high` - Major feature broken
- `p2-medium` - Minor feature issue  
- `p3-low` - Nice to have

### Size
- `size/S` - < 4 hours
- `size/M` - 1-2 days
- `size/L` - 3-5 days
- `size/XL` - > 1 week

### Type
- `feature` - New functionality
- `bug` - Something broken
- `enhancement` - Improvement to existing feature
- `docs` - Documentation only
- `refactor` - Code improvement, no behavior change

### Component
- `api` - FastAPI endpoints
- `agents` - DSPy agents  
- `database` - Models and migrations
- `frontend` - UI components
- `ci-cd` - Pipeline and deployment

### Status
- `needs-triage` - Needs review
- `ready` - Definition of Ready met
- `blocked` - Waiting on dependency
- `in-progress` - Being worked on

## 🔄 Workflow

1. **New Issue Created**
   - Automatically labeled `needs-triage`
   - Goes to Backlog column

2. **Triage**
   - Product owner reviews
   - Adds priority and size labels
   - Ensures Definition of Ready

3. **Development**
   - Developer self-assigns
   - Creates feature branch
   - Updates issue with progress

4. **Review**
   - Opens PR with "Fixes #123"
   - Automated checks run
   - Code review required

5. **Merge & Deploy**
   - PR merged to main
   - CI/CD pipeline triggered
   - Issue auto-closed

## 📈 Metrics to Track

- **Cycle Time**: Ready → Done
- **Lead Time**: Created → Done  
- **WIP Limit**: Max 3 issues In Development per developer
- **Bug Rate**: Bugs created vs features shipped

## 🤖 GitHub Actions Integration

The pipeline automatically:
- Runs tests on PR
- Updates issue status
- Deploys on merge
- Posts deployment status