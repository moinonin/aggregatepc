# AggregatePC Specification

## Project Title: Local Distributed Heterogeneous Compute Network

**Objective:** Interconnect local machines to perform distributed LLM inference and general-purpose parallel computing with automatic model discovery and decentralized compute.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    CONTROLLER MACHINE                      │
│                                                           │
│  aggregatepc controller                                   │
│  ├── Reads configs/cluster.conf                           │
│  ├── Accepts worker joins (UDP :8765)                     │
│  ├── Tracks worker health (heartbeat)                     │
│  ├── Dispatches tasks to best-fit nodes                   │
│  ├── Supports split model placement                       │
│  └── Web status query (UDP :8765)                         │
│                                                           │
│  Output:                                                  │
│  [aggregatepc] + Worker joined: defi (defi)               │
│    - CPU: 24c, RAM: 64196MB, GPU: RTX 3050               │
│    [score: 702.7] Models: llama-7b, mistral-7b            │
│  [aggregatepc] + Worker joined: nr-dell (nr-dell)         │
│    - CPU: 8c, RAM: 15879MB, GPU: Quadro P600              │
│    [score: 95.5] Models: none                             │
└──────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Worker 1   │ │   Worker 2   │ │   Worker 3   │
     │  defi        │ │  nr-dell     │ │  (future)    │
     │              │ │              │ │              │
     │ Auto-detect: │ │ Auto-detect: │ │              │
     │ • Hardware   │ │ • Hardware   │ │              │
     │ • Models     │ │ • Models     │ │              │
     │ • Ollama     │ │ • Ollama     │ │              │
     └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Decentralized Compute

### Model Placement Strategies

1. **Single GPU** — Model fits on one GPU (fastest)
2. **Multi-GPU Split** — Model split across multiple GPUs (when no single GPU has enough VRAM)
3. **CPU Fallback** — Model runs on CPU when no GPU has enough VRAM (slower but functional)
4. **Hybrid GPU+CPU** — GPU layers + CPU offload (uses all available resources)

### Split Model Allocation

When a model doesn't fit on any single GPU, the allocator distributes it:

```
Example: 70B model (14GB FP16) on cluster:
  - defi (RTX 3050 8GB):     layers 0-20   (8GB VRAM)
  - defi (System RAM 64GB):  layers 21-40  (24GB offload)
  - nr-dell (System RAM 16GB): layers 41-60 (12GB offload)
  - nr-dell (Quadro P600):   layers 61-70  (4GB VRAM)
```

### Task Retry with Backoff

Tasks that can't be scheduled immediately enter a retry queue:

```
Attempt 1: immediate
Attempt 2: after 5s
Attempt 3: after 10s
Attempt 4: after 20s (final)
```

---

## Model Discovery

### Supported Sources

| Source | Location | Type |
|--------|----------|------|
| HuggingFace Cache | `~/.cache/huggingface/hub` | `huggingface` |
| Ollama Models | `~/.ollama/models` | `ollama` |
| llama.cpp GGUF | `~/models`, `~/.local/share/models` | `llama.cpp` |
| Custom Paths | `$GGML_MODELS`, `$HF_HOME` | varies |

### Best Model Selection

Priority order:
1. **Ollama models** (already served, fastest to use)
2. **Models fitting in VRAM** (largest first)
3. **CPU fallback** (largest model that fits in RAM)

---

## Configuration

### configs/cluster.conf

```ini
[controller]
192.168.1.4

[worker]
192.168.100.11
192.168.100.31

[ports]
# controller_port = 8765
# status_port = 8766
```

### Resolution Order

1. CLI flags (highest priority)
2. Config file
3. Auto-discovery (mDNS)
4. Defaults

---

## CLI Commands

```bash
# Start controller
aggregatepc controller
make controller

# Start worker (auto-discover controller)
aggregatepc worker
make worker

# Join specific controller
aggregatepc worker --controller 192.168.1.4

# Profile hardware
aggregatepc profile
aggregatepc profile --scan

# Check cluster status
aggregatepc status
make status
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make help` | Show all targets |
| `make controller` | Start as cluster controller |
| `make worker` | Start as worker node |
| `make worker CONTROLLER=IP` | Join specific controller |
| `make profile` | Profile local hardware |
| `make profile SCAN=1` | Profile + network scan |
| `make status` | Show cluster status |
| `make test` | Run test suite |
| `make clean` | Remove generated files |
| `make install` | pip install -e . |

---

## Status Output

```json
{
  "workers": [
    {
      "node_id": "defi",
      "role": "worker",
      "status": "idle",
      "address": "192.168.100.11",
      "hardware": {
        "hostname": "defi",
        "cpu_name": "AMD Ryzen 9 5900X 12-Core Processor",
        "cpu_cores": 24,
        "ram_mb": 64196,
        "gpus": [
          {"name": "NVIDIA GeForce RTX 3050", "vram_mb": 8192, "integrated": false}
        ]
      },
      "compute_score": 702.69,
      "models": ["llama-7b", "mistral-7b"]
    }
  ],
  "worker_count": 2,
  "available_count": 2,
  "cluster_metrics": {
    "total_cpu_cores": 32,
    "total_ram_mb": 80075,
    "total_ram_gb": 78.2,
    "total_vram_mb": 8192,
    "total_vram_gb": 8.0,
    "total_models": 2,
    "available_models": ["llama-7b", "mistral-7b"]
  }
}
```

---

## Tests

```bash
make test
# 31 tests covering:
# - Hardware detection
# - Network discovery
# - Cluster formation (worker join)
# - Task queue (priority, dedup, completion)
# - GPU allocator (single fit, split, CPU fallback)
# - Scheduler (assignment, retry, insufficient resources)
# - CLI (help, subcommands)
# - Config loading
# - Model discovery
```

---

## Roadmap

- [x] Sprint 1: Hardware auto-detection
- [x] Sprint 2: Cluster formation & heartbeat
- [x] Sprint 3: Task scheduling & VRAM-aware allocation
- [x] Sprint 4: Config, CLI, packaging
- [x] Sprint 5: Model discovery & Ollama
- [x] Sprint 6: Decentralized compute (split placement, CPU fallback, retry)
- [ ] Sprint 7: Production hardening, web dashboard, encryption