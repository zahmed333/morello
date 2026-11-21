"""Where a monitor's saved state lives: a local folder, cloud storage, or a warehouse table.

A store maps a name to bytes. One job at a time should write a given name: stores do not lock,
and the last writer wins.
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from .sql import Source

EXTENSION = ".json.gz"
CHUNK_CHARS = 700_000  # base64 characters per row in a table store


class Store(Protocol):
    def put(self, name: str, data: bytes) -> None: ...

    def get(self, name: str) -> bytes | None:
        """The bytes last put under `name`, or None when nothing was."""
        ...


class LocalStore:
    """Files in a folder. A write replaces the file in one step, so no reader sees half of one."""

    def __init__(self, root: str) -> None:
        self.root = str(root)

    def put(self, name: str, data: bytes) -> None:
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, name + EXTENSION)
        temporary = path + ".tmp"
        with open(temporary, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)

    def get(self, name: str) -> bytes | None:
        path = os.path.join(self.root, name + EXTENSION)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as file:
            return file.read()


class FsspecStore:
    """Any fsspec URL: s3://bucket/path, abfss://..., gs://..., dbfs:/...

    Object stores replace an object in one step, so a reader sees the old or the new version.
    """

    def __init__(self, root: str) -> None:
        import fsspec

        self.filesystem, self.root = fsspec.core.url_to_fs(str(root))

    def put(self, name: str, data: bytes) -> None:
        self.filesystem.makedirs(self.root, exist_ok=True)
        with self.filesystem.open(f"{self.root}/{name}{EXTENSION}", "wb") as file:
            file.write(data)

    def get(self, name: str) -> bytes | None:
        path = f"{self.root}/{name}{EXTENSION}"
        if not self.filesystem.exists(path):
            return None
        with self.filesystem.open(path, "rb") as file:
            return file.read()


class TableStore:
    """State kept in a warehouse table, so scheduled jobs anywhere can share it.

    The table is append-only. Every put writes a new version as one or more rows, and get reads
    the highest complete version. Nothing is updated or deleted, which works on every engine,
    and earlier versions stay available for inspection. Versions count up from the highest one
    present, so their order does not depend on any machine's clock.
    """

    def __init__(self, source: Source, table: str) -> None:
        self.source, self.table = source, table

    def put(self, name: str, data: bytes) -> None:
        text = base64.b64encode(data).decode("ascii")
        chunks = [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)] or [""]
        rows = pd.DataFrame(
            {
                "name": name,
                "version": self._latest_version(name) + 1,
                "part": range(len(chunks)),
                "n_parts": len(chunks),
                "chunk": chunks,
            }
        )
        self.source.append(self.table, rows)

    def get(self, name: str) -> bytes | None:
        for version in range(self._latest_version(name), 0, -1):
            rows = self.source.query(
                f"SELECT part, n_parts, chunk FROM {self.table} "
                f"WHERE name = {_literal(name)} AND version = {version} ORDER BY part"
            )
            rows.columns = [column.lower() for column in rows.columns]
            if len(rows) and len(rows) == int(rows["n_parts"].iloc[0]):
                return base64.b64decode("".join(rows.sort_values("part")["chunk"]))
            # otherwise a writer is part-way through this version: use the one before it
        return None

    def _latest_version(self, name: str) -> int:
        """The highest version written under `name`, or 0 when there is none."""
        if not self.source.table_exists(self.table):
            return 0
        result = self.source.query(
            f"SELECT MAX(version) AS v FROM {self.table} WHERE name = {_literal(name)}"
        )
        latest = result.iloc[0, 0]
        return 0 if pd.isna(latest) else int(latest)


def open_store(target: str | Store) -> Store:
    """A store object as is; a URL as an FsspecStore; anything else as a local folder."""
    if hasattr(target, "put") and hasattr(target, "get"):
        return target
    location = str(target)
    if "://" in location or location.startswith("dbfs:/"):
        return FsspecStore(location)
    return LocalStore(location)


def _literal(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"
