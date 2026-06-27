# Cluster Configuration Specification (Phase 1)

## Project Title: Local Distributed Heterogeneous Compute Network

**Objective:** Interconnect 3 local machines to perform distributed LLM inference and general-purpose parallel computing.

---

## 🛠️ Execution Manual: Hardware & Network Audit

Copy this entire specification into your project folder. Run the diagnostic commands below on each machine, record the outputs directly into the markdown tables, and verify the baseline network connections.

---

## 1. Network Topology & Routing Map

Run the appropriate network lookup command on each machine to identify its local IPv4 address.

| Platform | Command |
|----------|---------|
| **Windows** | Open PowerShell and run: `Get-NetIPAddress -AddressFamily IPv4 \| Where-Object InterfaceMetric -le 25 \| Select-Object IPAddress` |
| **Linux** | Open Terminal and run: `hostname -I` |
| **macOS** | Open Terminal and run: `ipconfig getifaddr en0` |

### Local Node Routing Table

Record your network layout here once discovered:

| Node ID | Hostname / Role | Operating System | Connection Type | Local IP Address |
|---------|-----------------|------------------|-----------------|----------------|
| Node 1 | Primary Controller | Wired Ethernet | 192.168. | |
| Node 2 | Worker 1 | Wired Ethernet | 192.168. | |
| Node 3 | Worker 2 | macOS | Wi-Fi (5GHz/6) | 192.168. | |

---

## 2. Hardware Profile Diagnostics

Execute the system diagnostic queries on each designated machine to audit the CPU, System RAM, GPU architecture, and Available VRAM.

### 🖥️ Windows Node Diagnostic Commands

Run these commands inside PowerShell:

```powershell
# Query CPU Model and System RAM Size (GB)
Get-CimInstance Win32_Processor | Select-Object Name; [Math]::Round((Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum / 1GB, 2)

# Query GPU Model Name and Dedicated VRAM Pool
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
```

> **Note:** If the Windows node utilizes an NVIDIA GPU, run `nvidia-smi` to quickly verify exact VRAM allocations.

### 🐧 Linux Node Diagnostic Commands

Run these commands inside Bash Terminal:

```bash
# Query CPU Model and System RAM Size (GB)
lscpu | grep "Model name"; free -g

# Query GPU Architecture and VRAM Allocations
nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 🍏 macOS Node Diagnostic Commands

Run these commands inside Zsh Terminal:

```bash
# Query Apple Silicon Chip Family and Total Unified Memory
system_profiler SPHardwareDataType | grep -E "Chip|Memory"

# Query Core Profiles (CPU and Integrated Metal GPU Cores)
system_profiler SPDisplaysDataType | grep -E "Chipset Model|Total Number of Cores"
```

### Hardware Inventory Log

Paste the resulting values from your diagnostic outputs into this table:

| Node ID | CPU Architecture | System RAM | GPU Brand & Model | Video Memory (VRAM / Unified) |
|---------|------------------|------------|-------------------|----------------------------|
| Node 1 | | | | |
| Node 2 | | | | |
| Node 3 | Integrated Apple Silicon | | | |
| | **TOTALS** | **Aggregate Cores:** | **Total System RAM:** | **Cluster Engines:** | **Total Cluster VRAM:** | |

---

## 3. Network Interconnectivity & Firewall Verification

Before installing framework dependencies, verification of uninhibited peer-to-peer data transmission over the local router is required.

### Protocol Verification Sequence

**ICMP Echo Route Test:** From Node 3 (Mac), initiate a network trace to Node 1 (Primary):

```bash
ping -c 4 [Insert_Node_1_IP]
```

**Reverse Route Test:** From Node 1 (Primary), initiate a network trace back to Node 3 (Mac):

```bash
ping [Insert_Node_3_IP]
```

### Diagnostic Validation Checklist

- [ ] Node 3 → Node 1: Successful packets transmitted with 0% packet loss
- [ ] Node 1 → Node 3: Successful packets transmitted with 0% packet loss
- [ ] Latency Delta: Round-trip-time (rtt) latency variance is stable (typically under 10 ms)

### ⚠️ Troubleshooting Protocol: Firewall Drops

If a ping command returns "Request timed out" or "100% packet loss", the host machine's firewall is dropping local packets. Fix this before continuing:

**Windows:** Open PowerShell as Administrator and run:

```powershell
Set-NetConnectionProfile -InterfaceAlias (Get-NetAdapter | Where-Object Status -eq "Up").Name -NetworkCategory Private
```

> This reconfigures your network connection from "Public" to "Private", opening safe local network sharing ports.

---

## 🚀 Next Steps Summary

1. Save this document as `SPEC.md` inside your local workspace
2. Complete the diagnostic inputs in Section 1 and Section 2
3. Confirm the connectivity checklist in Section 3
4. Once you have filled out these baseline specifications, provide the data logs here. We will use them to compute the maximum model parameter bounds (e.g., 8B, 14B, 32B, or 70B models) your cluster can process, choose the quantization target (FP16, INT8, INT4), and write the orchestration configuration file!