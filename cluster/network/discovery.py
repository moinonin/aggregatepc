"""Local network discovery for automatic cluster formation."""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiscoveredPeer:
    """A peer discovered on the local network."""
    service_name: str
    address: str
    port: int
    properties: dict[str, str] = field(default_factory=dict)


class MDNSDiscovery:
    """Discover aggregatepc peers via mDNS/Bonjour on the local network."""

    SERVICE_TYPE = "_aggregatepc._tcp.local."
    MDNS_ADDR = "224.0.0.251"
    MDNS_PORT = 5353

    def __init__(self, port: int = 8765):
        self.port = port
        self._peers: list[DiscoveredPeer] = []
        self._running = False
        self._listener: Optional[threading.Thread] = None

    def get_local_ip(self) -> str:
        """Get the local IPv4 address used for outbound connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    def scan_peers(self, timeout: float = 3.0) -> list[DiscoveredPeer]:
        """Scan the local network for aggregatepc peers via mDNS.

        Returns a list of discovered peers. Requires the `zeroconf` package.
        Falls back to simple UDP broadcast scan if zeroconf is not available.
        """
        try:
            return self._scan_zeroconf(timeout)
        except ImportError:
            return self._scan_broadcast(timeout)

    def _scan_zeroconf(self, timeout: float) -> list[DiscoveredPeer]:
        """Scan using the zeroconf library (preferred method)."""
        from zeroconf import ServiceBrowser, Zeroconf, ServiceStateChange

        zeroconf = Zeroconf()
        peers: list[DiscoveredPeer] = []
        event = threading.Event()

        def add_service(zc: Zeroconf, service_type: str, name: str) -> None:
            info = zc.get_service_info(service_type, name)
            if info:
                addresses = info.parsed_addresses()
                if addresses:
                    peers.append(DiscoveredPeer(
                        service_name=name,
                        address=addresses[0],
                        port=info.port,
                        properties={k.decode(): v.decode() for k, v in info.properties.items()},
                    ))
                    event.set()

        browser = ServiceBrowser(zeroconf, self.SERVICE_TYPE, handlers=[ServiceStateChange(add=add_service)])
        event.wait(timeout=timeout)
        zeroconf.close()
        return peers

    def _scan_broadcast(self, timeout: float) -> list[DiscoveredPeer]:
        """Fallback: scan via UDP broadcast on the local subnet."""
        peers: list[DiscoveredPeer] = []
        local_ip = self.get_local_ip()
        subnet = ".".join(local_ip.split(".")[:3]) + ".255"

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(timeout)

            # Send discovery probe
            probe = b"AGGREGATEPC_DISCOVER"
            s.sendto(probe, (subnet, self.port + 1))

            # Listen for responses
            try:
                while True:
                    data, addr = s.recvfrom(1024)
                    if data.startswith(b"AGGREGATEPC_HERE"):
                        peers.append(DiscoveredPeer(
                            service_name=data.decode().split(":", 1)[1] if b":" in data else "unknown",
                            address=addr[0],
                            port=self.port,
                        ))
            except socket.timeout:
                pass

        return peers

    def advertise(self) -> None:
        """Start advertising this node on the local network."""
        try:
            self._advertise_zeroconf()
        except ImportError:
            self._advertise_broadcast()

    def _advertise_zeroconf(self) -> None:
        """Advertise via zeroconf."""
        from zeroconf import Zeroconf, ServiceInfo

        info = ServiceInfo(
            self.SERVICE_TYPE,
            f"{socket.gethostname()}.{self.SERVICE_TYPE}",
            port=self.port,
            properties={"version": "0.1.0", "role": "worker"},
        )
        zeroconf = Zeroconf()
        zeroconf.register_service(info)
        self._running = True

    def _advertise_broadcast(self) -> None:
        """Advertise via UDP broadcast listener."""
        self._running = True
        self._listener = threading.Thread(target=self._broadcast_listener, daemon=True)
        self._listener.start()

    def _broadcast_listener(self) -> None:
        """Listen for discovery probes and respond."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", self.port + 1))
            s.settimeout(1.0)

            while self._running:
                try:
                    data, addr = s.recvfrom(1024)
                    if data == b"AGGREGATEPC_DISCOVER":
                        response = f"AGGREGATEPC_HERE:{socket.gethostname()}".encode()
                        s.sendto(response, addr)
                except socket.timeout:
                    continue
                except OSError:
                    break

    def stop(self) -> None:
        """Stop advertising and scanning."""
        self._running = False
        if self._listener:
            self._listener.join(timeout=2.0)
            self._listener = None


def discover_peers(port: int = 8765, timeout: float = 3.0) -> list[DiscoveredPeer]:
    """Convenience function: scan for aggregatepc peers on the local network."""
    discovery = MDNSDiscovery(port=port)
    return discovery.scan_peers(timeout=timeout)
