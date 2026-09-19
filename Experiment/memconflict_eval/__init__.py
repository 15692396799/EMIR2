"""MemConflict experiment harness for the Retrival-Mem (AutoRetri) memory system.

This package drives the *unmodified* Retrival-Mem checkout as the memory system
under test, and the *unmodified* MemConflict dataset as the benchmark. Nothing
under ``Retrival-Mem/`` or ``MemConflict/`` is edited; both are used as
read-only dependencies.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
