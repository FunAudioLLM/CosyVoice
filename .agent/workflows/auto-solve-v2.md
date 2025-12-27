---
description: v2 - Systematically and recursively solve complex tasks with full autonomy.
---

# Antigravity Workflow: Auto-Solve v2 (Issue/PR Driven)

This workflow is a **repeatable process** to take a user request from “idea” to “merged” with strong safety guarantees.

## Workflow Constraints

- **Workflow file size:** keep this workflow ≤ **12,000 characters**.
- **One session, one parent:** all work belongs to a single session parent issue.
- **One issue, one PR:** every atomic task (sub-issue) maps to exactly one PR.

## 1) Session Identity & Scope

### Session ID
`YYYYMMDD-HHMMSS-<shortsha>`

### Mandatory label
- `agent:session:<SessionID>`

## 2) Required Workspace Artifacts

```
.agent/
  request.md
  plan.md
  backlog.md
  session.json
  logs/run.log
```

## 3) Logging & Command Safety

- Log every command (timestamp + exit code).
- Use `set -euo pipefail`.
- Never log secrets.

## 4) Request → Plan → Backlog

- Capture request verbatim.
- Produce plan with requirements, risks, tests.
- Split into atomic backlog items (1 PR each).

## 5) GitHub Issue Hierarchy

- Parent issue per session.
- Sub-issue per backlog item.
- Link issues via GraphQL when possible.

## 6) Work Queue & Recursion

Process open sub-issues until none remain.
Split oversized tasks into smaller ones.

## 7) Branch / PR Discipline

- Branch: `agent/<SessionID>/issue-<IssueNumber>-<slug>`
- Run quality gates before PR.
- Merge, delete branch, verify closure.

## 8) Failure Handling

- Fix or explicitly block.
- No silent skips.

## 9) End-State Cleanup

- Reset to default branch.
- Clean repo.
- Delete session branches.
- Close parent issue.

## 10) Definition of Done

- All sub-issues closed.
- All PRs merged.
- Repo clean.
