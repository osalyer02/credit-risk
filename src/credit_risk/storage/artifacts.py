"""Local filesystem artifact persistence primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    """Common interface for artifact persistence backends."""

    def put_bytes(self, key: str, data: bytes) -> str:
        """Store bytes and return a fully qualified location."""

    def get_bytes(self, key: str) -> bytes:
        """Load bytes at key."""


@dataclass
class LocalArtifactStore:
    """File-system implementation of artifact persistence."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_write(self, key: str) -> Path:
        path = self.root_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _path_for_read(self, key: str) -> Path:
        return self.root_dir / key

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path_for_write(key)
        path.write_bytes(data)
        return str(path)

    def get_bytes(self, key: str) -> bytes:
        path = self._path_for_read(key)
        return path.read_bytes()
