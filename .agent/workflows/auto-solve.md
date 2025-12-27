---
description: Systematically and recursively solve complex tasks with full autonomy.
---

# Autonomous Task Solver

You are in **Autonomous Mode**. Follow this workflow strictly.

> [!IMPORTANT]
> **CRITICAL RULES**:
> 1.  **Log EVERYTHING**: Use `log_cmd` for all commands.
> 2.  **Verify**: Never mark done without `tsc`, `lint`, and `test`.
> 3.  **Strict GitHub**: Always use Issues, Branches, and PRs.

---

## Phase 0: Setup & Planning
// turbo-all

### 0.1 Initialize Session
```bash
# Setup Environment & Logging
export TERM=xterm-256color
set -o pipefail
SessionID=${SessionID:-"auto-$(date +%s)"}
LogFile=".agent/sessions/$SessionID/session.log"
mkdir -p "$(dirname "$LogFile")"
exec > >(tee -a "$LogFile") 2>&1

# Logging Helpers
log_cmd() { echo -e "\n\033[1;33m▶ EXECUTE:\033[0m $*"; "$@"; }
log_info() { echo -e "\033[0;36m$*\033[0m"; }
log_succ() { echo -e "\033[0;32m✓ $*\033[0m"; }
log_err()  { echo -e "\033[0;31m✗ $*\033[0m"; }

log_info "=== Session Started: $SessionID ==="

### 0.2 Repo & Issue State Prep
if [ -n "$(git status --porcelain | head -1)" ]; then
    log_info "⚠️ Dirty working directory. Committing WIP..."
    log_cmd 'git add -A && git commit -m "wip: pre-agent state preservation" || true'
fi

# Ensure Remote & Labels
if ! git remote -v | grep -q origin; then
    RepoName=$(basename "$PWD")
    log_cmd "gh repo create \"$RepoName\" --private --source=. --remote=origin --push --enable-issues"
fi

Labels=("agent:auto|000000" "status:in-progress|1d76db")
for l in "${Labels[@]}"; do
    IFS='|' read -r n c <<< "$l"
    gh label create "$n" -c "$c" --force || true
done

# Create Parent Issue
if [ -z "$ParentID" ]; then
    ParentTitle="Agent Session: $SessionID"
    # Parse URL for ID to avoid flag compatibility issues
    ParentURL=$(gh issue create --title "$ParentTitle" --body "Tracking issue for session $SessionID" --label "agent:auto" | grep "http") || true
    ParentID=$(echo "$ParentURL" | awk -F'/' '{print $NF}' | tr -d '\r')
    [ -z "$ParentID" ] && { log_err "Failed to create Parent Issue. Fallback to 1."; ParentID=1; }
    log_succ "Parent Issue Created: #$ParentID"
else
    log_info "Resuming Parent Issue: #$ParentID"
fi
```

### 0.3 Task Decomposition
### 0.3 Task Decomposition
**Action**: Parse `task.md` and create sub-issues.
> [!TIP]
> **Refactoring Safety**: If a task involves refactoring:
> 1. `grep` for all usages of the target code.
> 2. List all capabilities (e.g. methods, props) that must be preserved.
> 3. Verify the new design handles all existing use cases.

```bash
set -o pipefail
exec > >(tee -a "$LogFile") 2>&1

create_task_issue() {
    local t_name="$1"
    local existing=$(gh issue list --search "$t_name in:title" --state open --json number -q '.[0].number')
    if [ -n "$existing" ]; then
        echo "$existing"
    else
        gh issue create --title "$t_name" --body "Section of #$ParentID" --label "agent:auto" | grep "http" | awk -F'/' '{print $NF}' | tr -d '\r'
    fi
}
# Usage: IssueID=$(create_task_issue "Task Name")
```

---

## Phase 1: Execution Loop

**COMMAND: Process ALL Issues**
Execute until no open sub-issues remain linked to the Parent Issue.

```bash
set -o pipefail
exec > >(tee -a "$LogFile") 2>&1

get_next_issue() {
  gh issue list --label "agent:auto" --state open --json number --jq '.[] | select(.number != '$ParentID') | .number' | head -1
}

while true; do
  IssueID=$(get_next_issue)
  [ -z "$IssueID" ] && { log_succ "All tasks completed!"; break; }

  log_info "=== Starting Task #$IssueID ==="

  # 1.1 Branch Management
  BranchName="agent/task-$IssueID"
  if [ "$(git branch --show-current)" != "$BranchName" ]; then
      if git show-ref --verify --quiet "refs/heads/$BranchName"; then
          log_cmd "git checkout $BranchName"
      else
          log_cmd "gh issue develop $IssueID --checkout --name $BranchName" || log_cmd "git checkout -b $BranchName"
      fi
  fi

  # 1.2 Implement & Verify
  # [ACTION REQUIRED]: READ Issue -> EDIT Code -> VERIFY
  # Loop manually until passing:
  # Loop manually until passing:
  log_cmd "pnpm exec tsc --noEmit" || {
      log_err "Type Errors";
      # If stuck on generic/library types, search for similar patterns in codebase or read docs
      continue;
  }
  log_cmd "pnpm lint" || { log_err "Lint Errors"; continue; }
  log_cmd "pnpm test" || { log_err "Test Failures"; continue; }

  # 1.3 Commit & Push
  log_cmd 'git add -A'
  log_cmd 'git commit -m "feat: implement task #'$IssueID' (agent)"'
  log_cmd 'git push -u origin HEAD'

  # 1.4 PR & Merge
  PRUrl=$(gh pr list --head "$BranchName" --json url -q '.[0].url')
  if [ -z "$PRUrl" ]; then
      PRUrl=$(gh pr create --title "feat: Task #$IssueID" --body "Closes #$IssueID" --label "agent:auto" | grep "http")
  fi
  log_cmd "gh pr merge $PRUrl --merge --delete-branch" || log_err "Merge failed, verify manually."

  # 1.5 Close Issue & Reset
  log_cmd "gh issue close $IssueID"
  DefaultBranch=$(git remote show origin | grep 'HEAD branch' | cut -d' ' -f5)
  log_cmd "git checkout $DefaultBranch && git pull"

  log_succ "Task #$IssueID Complete"
done
```

---

## Phase 2: Completion

### 2.1 Final Verification
```bash
set -o pipefail
exec > >(tee -a "$LogFile") 2>&1
log_info "=== Final System Check ==="
log_cmd "pnpm build" && log_succ "Build Verified"
```

### 2.2 Close Session
```bash
set -o pipefail
exec > >(tee -a "$LogFile") 2>&1
gh issue close $ParentID --comment "Session $SessionID completed successfully."
log_succ "Session Closed"
```

### 2.3 Documentation
- **Update**: `CHANGELOG.md`
- **Create**: `walkthrough.md`