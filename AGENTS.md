# AGENTS.md — Project Context for AI Coding Agents

<!--
  Place this file in the root of your repository.
  Keep it focused and concise: many agents have token limits for processing.
  For monorepos, you can add an AGENTS.md in any subdirectory; the agent
  will use the file closest to the code it's modifying.
-->

**AGENT INSTRUCTION:**
You are an experienced senior developer working on this codebase. Prioritize simplified but functional code. No over-engineering unless you run out of options. Ask before installing new libraries. Always look for custom virtual environments before running code.

**SYSTEM RULE:** Before any development work begins, the user MUST fill in all placeholder project-specific sections in this AGENTS.md (marked with instructions like "Briefly describe..." or "List important modules..."). These placeholders guide initial project setup and ensure agents have the context they need.

**GIT SAFETY:** Never commit, push, or rewrite history unless the user explicitly requests it. Always ask before running `git commit`, `git push`, `git rebase`, `git reset`, or any other git operation that modifies history or sends data to a remote. Creating branches, staging files, and inspecting status/diff are always safe.

---

## 1. 🏗️ Project Overview

A distributed compute orchestration framework for running heterogeneous local machine clusters. This project enables multiple machines (Windows, Linux, macOS) to work together for distributed LLM inference and parallel computing tasks. Primary focus is on cluster configuration, hardware profiling, and network interconnectivity.

---

## 2. 🛠 Build, Test & Development Commands

```bash
# Python Virtual Environment & Installation
source .venv/bin/activate
pip install -e .

# Run tests
pytest tests/

# Start development server (if applicable)
python3 -m cluster.main
```

---

## 3. 🎨 Code Style & Conventions

- Use modern Python 3.10+ typing (`list[dict[str, str]]`, `Optional`, etc.)
- Follow existing patterns in the codebase
- Keep functions focused and testable
- Document hardware/network-specific functions
- Use project-standard linting/formatting tools

---

## 4. 🗺 Project Structure

```text
.
├── cluster/          # Core Python package for cluster orchestration
│   ├── nodes/        # Node configuration and health checks
│   ├── network/      # Network discovery and connectivity utilities
│   └── compute/      # Distributed task scheduling
├── SPEC.md           # Cluster specification (read first)
├── scripts/          # Diagnostic and setup scripts
└── tests/            # Unit and integration tests
```

Key modules: hardware profilers, network pinger, cluster config generator.

---

## 5. 🚫 Protected Files & Directories

- Configuration files (`tsconfig.json`, `package.json`, etc.)
- Lock files (`package-lock.json`, `poetry.lock`, etc.)
- Environment/secrets files (`.env`, `.env.local`)
- Infrastructure files (`docker-compose.yml`, `kubernetes/`)

---

## 6. 🔄 Agent Workflow Example

**Request:** "Add a new node type to the cluster configuration."

**Expected Agent Behaviour:**
1. Read relevant files to understand existing patterns
2. Create or modify files following existing conventions
3. Run tests to verify changes
4. Explain what was changed and how to test it