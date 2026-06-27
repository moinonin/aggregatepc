# Project Sprints & Implementation Plan

**Project Goal:** Build an automated GPU/CPU aggregation system that enables users with free PCs to pool their idle computing resources for distributed LLM inference and general-purpose parallel computing through a simple, user-friendly interface.

---

## Sprint 1: Core Infrastructure & Auto-Detection ✅

### Objective
Enable automatic detection and profiling of local hardware resources without manual configuration.

### Completed
- [x] `cluster/detect.py` - Auto-detect CPU cores, RAM, GPU models, and VRAM across platforms
- [x] `cluster/network/discovery.py` - Automatic local network discovery using mDNS/SSDP
- [x] `scripts/auto_profile.py` - One-click hardware and network profiling script
- [x] `cluster/nodes/__init__.py` - Node abstraction layer (Windows, Linux, macOS)

### Platform-Specific Handling
- Windows: WMI queries for CPU/GPU, psutil for memory
- Linux: lscpu, nvidia-smi, /proc/meminfo parsing
- macOS: system_profiler, unified memory reporting

---

## Sprint 2: Cluster Formation & Bootstrap ✅

### Objective
Create a zero-config cluster formation process where users can claim "worker" status on their machines.

### Completed
- [x] `cluster/nodes/worker.py` - Lightweight worker daemon for idle PCs
- [x] `cluster/nodes/controller.py` - Controller that aggregates workers
- [x] `scripts/start_worker.py` - Simple script to turn any PC into a worker
- [x] `cluster/network/heartbeat.py` - Worker health monitoring and auto-reconnect
- [x] CLI: `aggregatepc worker` - Auto-start when PC is idle

### User Story
> Users run `aggregatepc worker` on their old laptops/desktops. The script auto-detects hardware, joins the local network cluster, and waits for tasks.

---

## Sprint 3: Task Distribution & Load Balancing ✅

### Objective
Intelligently distribute compute tasks across heterogeneous nodes based on their capabilities.

### Completed
- [x] `cluster/compute/scheduler.py` - Capability-based task assignment
- [x] `cluster/compute/task_queue.py` - Priority queue with resource requirements
- [x] `cluster/compute/gpu_allocator.py` - VRAM-aware model placement
- [x] `cluster/tasks/llm_inference.py` - LLM inference task type
- [x] `cluster/tasks/batch_compute.py` - General batch processing task type

### Key Features
- Auto-split large models across nodes with available VRAM
- Fall back to CPU when GPU memory insufficient
- Task priority based on urgency and node availability

---

## Sprint 4: Configuration, CLI & Packaging ✅

### Objective
Provide a unified CLI, config file support, and installable packaging.

### Completed
- [x] `aggregatepc.py` - Unified CLI with controller/worker/profile/status commands
- [x] `cluster/config.py` - Config file loader (configs/cluster.conf)
- [x] `configs/cluster.conf` - IP addresses and ports
- [x] `Makefile` - Build targets (help, controller, worker, profile, status, test, clean)
- [x] `pyproject.toml` - pip install -e . support
- [x] `.gitignore` - Proper exclusions
- [x] `README.md` - Comprehensive documentation with code snippets

### Config Resolution Order
1. CLI flags (highest priority)
2. Config file (configs/cluster.conf)
3. Auto-discovery (mDNS scan)
4. Defaults (127.0.0.1:8765)

---

## Sprint 5: Model Discovery & Ollama Integration ✅

### Objective
Discover models already pulled on each node and auto-serve them via Ollama.

### Completed
- [x] `cluster/models/registry.py` - Discovers HF cache, Ollama, llama.cpp models
- [x] `cluster/models/ollama.py` - Full Ollama integration (detect, start, pull, list)
- [x] `cluster/nodes/__init__.py` - Added `models` field to Node dataclass
- [x] Worker advertises models on join
- [x] Controller prints model list in join notification
- [x] `get_best_model()` - Selects largest model fitting VRAM (prefers Ollama)
- [x] Worker auto-starts Ollama in background on startup

---

## Sprint 6: Decentralized Compute (Current)

### Objective
Enable true distributed compute by splitting models across nodes and falling back gracefully.

### Tasks
- [x] Split model placement across multiple GPUs/nodes
- [x] CPU offload fallback when VRAM insufficient
- [x] Task retry queue with exponential backoff
- [ ] Pipeline parallelism (layer-by-layer inference across nodes)
- [ ] Active model pulling on demand (controller triggers pull)
- [ ] Worker-to-worker communication for split inference

### Key Features
- **Split Placement**: When no single node has enough VRAM, the allocator distributes the model across multiple GPUs
- **CPU Fallback**: Models too large for GPU can run on CPU (slower but functional)
- **Retry Queue**: Tasks that can't be scheduled immediately are queued and retried with exponential backoff
- **Graceful Degradation**: Cluster remains useful even with limited resources

---

## Sprint 7: Production Hardening & Community Release

### Objective
Make the system stable for everyday use and easy for others to set up.

### Tasks
- [ ] Cross-platform installer script
- [ ] Auto-update mechanism for workers
- [ ] Security: encrypted task communication, token authentication
- [ ] Web dashboard for cluster monitoring
- [ ] Documentation: "Set up your old laptop in 5 minutes or less"
- [ ] Example configs for common use cases (LLM inference, video encoding, etc.)

---

## Implementation Principles

1. **Users First, Developers Second** - Every feature must pass the "grandma test" (easy enough for a non-developer to use)
2. **Idle-Only Policy** - Workers should never interfere when the user is actively using their PC
3. **Any PC Works** - Support Windows 10+, Linux, macOS with any hardware configuration
4. **No Cloud Required** - Everything runs on local network, no external dependencies
5. **Plug & Play** - Auto-discovery and zero-config setup whenever possible
6. **Decentralized Compute** - Models and tasks should be distributed across all available nodes