"""Hardware collectors."""

from .amd_linux import AmdLinuxCollector
from .apple import AppleCollector
from .nvidia import NvidiaCollector
from .system import SystemCollector, collect_specs

__all__ = ["AmdLinuxCollector", "AppleCollector", "NvidiaCollector", "SystemCollector", "collect_specs"]
