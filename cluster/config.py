"""Configuration loader for AggregatePC.

Reads cluster.conf (or a user-specified file) to discover controller and worker IPs.
Falls back to defaults when no config file is present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cluster.conf"

DEFAULT_CONTROLLER_IP = "127.0.0.1"
DEFAULT_WORKER_PORT = 8765


def load_config(config_path: Optional[str] = None) -> dict:
    """Load cluster configuration from a simple INI-style file.

    Returns a dict with:
        - controller_ip: str
        - worker_ips: list[str]
        - controller_port: int
        - status_port: int
        - proxy_host: str | None
        - proxy_port: int
        - relay_port: int
        - ollama_backends: dict[str, dict]
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    config = {
        "controller_ip": DEFAULT_CONTROLLER_IP,
        "worker_ips": [],
        "controller_port": DEFAULT_WORKER_PORT,
        "status_port": DEFAULT_WORKER_PORT + 1,
        "proxy_host": None,
        "proxy_port": 8000,
        "relay_port": 8767,
        "ollama_backends": {},
    }

    if not path.exists():
        return config

    current_section = None
    try:
        with open(path, "r") as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#") or line.startswith(";"):
                    continue

                # Section header
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip().lower()
                    continue

                # Key = value
                if "=" in line and current_section in ("ports", "proxy"):
                    key, _, value = line.partition("=")
                    key = key.strip().lower()
                    value = value.strip()
                    if current_section == "ports":
                        try:
                            if key == "controller_port":
                                config["controller_port"] = int(value)
                            elif key == "status_port":
                                config["status_port"] = int(value)
                            elif key == "proxy_port":
                                config["proxy_port"] = int(value)
                            elif key == "relay_port":
                                config["relay_port"] = int(value)
                        except ValueError:
                            print(f"[aggregatepc] Warning: invalid port value at line {line_num}: {line}")
                    elif current_section == "proxy":
                        if key == "host":
                            config["proxy_host"] = value
                        elif key == "port":
                            try:
                                config["proxy_port"] = int(value)
                            except ValueError:
                                print(f"[aggregatepc] Warning: invalid proxy port at line {line_num}: {line}")
                    continue

                if "=" in line and current_section in ("ollama", "backends"):
                    node_id, _, value = line.partition("=")
                    node_id = node_id.strip()
                    value = value.strip()
                    host, port = _parse_host_port(value, default_port=11434)
                    if node_id and host:
                        config["ollama_backends"][node_id] = {"host": host, "port": port}
                    continue

                # IP address line
                if current_section in ("controller", "worker"):
                    ip = line
                    # Basic validation
                    if "." in ip and not ip.startswith("#"):
                        if current_section == "controller":
                            config["controller_ip"] = ip
                        else:
                            config["worker_ips"].append(ip)

    except Exception as e:
        print(f"[aggregatepc] Warning: failed to read config from {path}: {e}")

    return config


def _parse_host_port(value: str, default_port: int) -> tuple[str, int]:
    """Parse host[:port] config values."""
    host = value
    port = default_port
    if ":" in value:
        host_part, _, port_part = value.rpartition(":")
        if host_part:
            host = host_part
        try:
            port = int(port_part)
        except ValueError:
            port = default_port
    return host.strip(), port


def save_default_config(config_path: Optional[str] = None) -> str:
    """Write a default config file if one doesn't exist."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    content = """# AggregatePC Cluster Configuration
#
# Register your machines here by role. The orchestration layer
# reads this file to know which IPs are available.
#
# Format:
#   - One IP per line under each section
#   - Lines starting with # are comments
#   - Leave a section empty if no machines serve that role

[controller]
# The machine that coordinates the cluster (only one)
192.168.1.5

[worker]
# Machines that contribute idle compute
192.168.1.10
192.168.1.11
# 192.168.1.12
# 192.168.1.13

[ports]
# Override default ports here if needed
# controller_port = 8765
# status_port = 8766
# proxy_port = 8000
# relay_port = 8767

[proxy]
# Optional: force the proxy bind address
# host = 192.168.1.5

[ollama]
# Optional: per-worker Ollama backend overrides, useful behind NAT/port forwarding
# worker-node-id = host:port
# nr-dell = 192.168.1.2:11435
# defi = 192.168.1.2:11436
"""
    path.write_text(content)
    return str(path)


if __name__ == "__main__":
    import json
    config = load_config()
    print(json.dumps(config, indent=2))
