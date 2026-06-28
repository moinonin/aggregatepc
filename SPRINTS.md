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

## Sprint 7: Inference Proxy & Cluster-Wide Model Routing ✅

### Objective
Enable any node to inference any model in the cluster, regardless of which node has the model locally. The proxy discovers models across the cluster and routes requests to the best available backend.

### Completed
- [x] `scripts/start_inference.py` - Cluster inference proxy with `--broadcast` mode
- [x] `ClusterProxy` class - Discovers models across workers via controller status + direct Ollama API checks
- [x] `ProxyHandler` - HTTP proxy that routes `/v1/chat/completions` and `/api/generate` to best node
- [x] Proxy queries controller status endpoint to get worker addresses and models
- [x] Proxy directly checks each worker's `/api/tags` for additional models
- [x] Auto-selects best model (largest, prefers Ollama, must be reachable)
- [x] Model auto-routing: client sends `"model":"any"` and proxy substitutes the best model
- [x] `/status` endpoint shows cluster-wide model availability
- [x] `/v1/models` endpoint lists available models (OpenAI-compatible)
- [x] Workers re-advertise models after Ollama starts (delayed heartbeat)
- [x] Heartbeat handler stores models from heartbeat messages
- [x] Makefile `make status` reads controller IP from config
- [x] README updated with correct inference commands and network configuration

### Architecture
```
Client → Proxy (:8000) → Controller status query → Worker Ollama (:11434)
         ↓
    Discovers best model across cluster
         ↓
    Routes request to worker that has the model
         ↓
    Returns response to client
```

### Key Features
- **No model pulling required** — Uses models already on workers
- **Cross-machine inference** — 8GB GPU machine can inference 70B model via cluster
- **Automatic failover** — Only selects from reachable backends
- **OpenAI-compatible API** — Works with any OpenAI client

---

## Sprint 8: External IP & WAN Support (Current)

### Objective
Enable the cluster to span multiple networks and external IP addresses, allowing workers in different locations to contribute compute resources.

### Tasks
- [ ] **Config-based external IP support** — Already works; just update `configs/cluster.conf` with external IPs
- [ ] **Port forwarding guide** — Document NAT traversal for workers behind routers
- [ ] **Relay/TURN server** — For workers behind restrictive NAT/firewalls
- [ ] **TLS encryption** — Encrypt all communications (controller ↔ worker, proxy ↔ Ollama)
- [ ] **Token authentication** — Secure cluster join and inference endpoints
- [ ] **STUN/TURN for NAT traversal** — Auto-detect and punch through NAT
- [ ] **Dynamic worker discovery** — DNS-based or DHT-based discovery without config files
- [ ] **Bandwidth-aware routing** — Prefer local workers for large models, remote for small
- [ ] **Connection multiplexing** — Single persistent connection per worker instead of UDP
- [ ] **Compression** — Compress model responses for WAN links

### Architecture for Multi-Network
```
Home Network                    Office Network
┌──────────────┐               ┌──────────────┐
│ Controller   │◄──────────────│ Worker C     │
│ :8765        │  WAN/VPN      │ External IP  │
│              │               │              │
│ Worker A     │               │ Worker D     │
│ 192.168.1.x  │               │ 10.0.0.x     │
│              │               │              │
│ Worker B     │               │              │
│ 192.168.1.x  │               │              │
└──────────────┘               └──────────────┘
```

### Implementation Plan

#### Phase 1: Network Infrastructure
1. Add `network_mode` to config (`lan` or `wan`)
2. For WAN mode, add `relay_address` for TURN server
3. Implement connection health checks with latency reporting
4. Add bandwidth estimation for routing decisions

#### Phase 2: Security
1. TLS certificate generation and distribution
2. Token-based authentication for worker join
3. API key authentication for inference proxy
4. Encrypted heartbeat messages

#### Phase 3: NAT Traversal
1. STUN client for public IP discovery
2. UDP hole punching for direct connections
3. TURN relay fallback for symmetric NAT
4. Auto-detection of NAT type

#### Phase 4: WAN Optimization
1. Response compression (gzip)
2. Connection pooling and keep-alive
3. Latency-based routing (prefer nearby workers)
4. Chunked transfer for large responses

### Configuration Additions
```ini
[network]
mode = lan                    # lan or wan
relay_address =               # TURN server address (WAN mode)
stun_server = stun.l.google.com:19302
tls_enabled = false

[security]
auth_token =                  # Cluster join token
api_key =                     # Inference API key
tls_cert =                    # Path to TLS certificate
```

---

## Sprint 9: Production Hardening & Community Release

### Objective
Make the system stable for everyday use and easy for others to set up.

### Tasks
- [ ] Cross-platform installer script
- [ ] Auto-update mechanism for workers
- [ ] Web dashboard for cluster monitoring
- [ ] Documentation: "Set up your old laptop in 5 minutes or less"
- [ ] Example configs for common use cases (LLM inference, video encoding, etc.)
- [ ] Docker/Podman support for containerized deployment
- [ ] Systemd/launchd service files for auto-start
- [ ] Health check endpoint for monitoring
- [ ] Prometheus metrics export
- [ ] CLI improvements (colored output, progress bars, interactive mode)

---

## Implementation Principles

1. **Users First, Developers Second** - Every feature must pass the "grandma test" (easy enough for a non-developer to use)
2. **Idle-Only Policy** - Workers should never interfere when the user is actively using their PC
3. **Any PC Works** - Support Windows 10+, Linux, macOS with any hardware configuration
4. **No Cloud Required** - Everything runs on local network, no external dependencies
5. **Plug & Play** - Auto-discovery and zero-config setup whenever possible
6. **Decentralized Compute** - Models and tasks should be distributed across all available nodes
7. **External IP Ready** - Cluster should span networks with minimal configuration