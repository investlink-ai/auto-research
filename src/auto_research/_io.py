"""Cross-package file-I/O primitives.

Today's exports are `atomic_write_bytes` and `atomic_write_text`. Both
write to disk via tmp → fsync(file) → rename → fsync(parent_dir) so a
power loss can't leave a torn or invisible result on the next boot.

Originally lived in `auto_research.ingest._http` for EDGAR's raw-bytes
persistence; `auto_research.extract.guardrails` now also needs the same
discipline for `data/quarantine/<worker>/<doc_id>.json` audit-trail
entries. Promoted here so both callers — and any future authoritative
write site (e.g., the manifest writer, extracted-output store) — share
one durability contract.

Scope is deliberately tiny: only the durable-write primitive. Anything
source- or worker-specific (path layout, tmp-naming policy beyond
collision-avoidance) stays at the call site.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path


def atomic_write_bytes(dest: Path, content: bytes) -> None:
    """Write `content` to `dest` atomically and durably.

    Discipline (mirrors `manifest.append`'s):

    - Write to a hidden tmp sibling, fsync the file, then `os.replace`
      onto the destination. The rename is atomic at the POSIX layer.
    - fsync the parent directory after the rename so the directory
      entry update is itself durable — without this, a power loss
      between rename and the implicit dir-flush could leave the new
      file invisible (or back at the old name) on recovery.
    - Tmp is removed on any failure between `write_bytes` and
      `os.replace` so a crash doesn't leak hidden dotfiles into the
      destination directory.
    - PID + thread-id in the tmp suffix prevents collisions when
      multiple `asyncio.to_thread` workers (or threads in a pool)
      happen to target the same dest.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_bytes(content)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, dest)
    except BaseException:
        # Cleanup must NOT mask the original exception. unlink can itself
        # raise (PermissionError on read-only fs, EIO on a dying disk) —
        # swallow OSError from cleanup and re-raise the original failure.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(dest: Path, content: str, *, encoding: str = "utf-8") -> None:
    """UTF-8 (by default) text variant of `atomic_write_bytes`.

    JSON audit records are UTF-8 by spec; `encoding` is a parameter only
    so a future caller writing latin-1-bound legacy data can opt out
    explicitly rather than re-implementing the durability discipline.
    """
    atomic_write_bytes(dest, content.encode(encoding))


__all__ = ["atomic_write_bytes", "atomic_write_text"]
