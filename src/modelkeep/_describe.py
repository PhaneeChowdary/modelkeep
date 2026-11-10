"""Describe a model file: its parts, their settings and the input that the model expects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._scan import Scan, Stub

MAX_SETTINGS = 6
MAX_DEPTH = 4


def _is_estimator(obj) -> bool:
    return isinstance(obj, Stub) and isinstance(obj.state, dict) and "_sklearn_version" in obj.state


def _settings(stub: Stub) -> str:
    shown = []
    for key, value in stub.state.items():
        if key.startswith("_") or key.endswith("_") or not isinstance(value, (str, int, float, bool, type(None))):
            continue
        shown.append(f"{key}={value!r}")
    more = f", +{len(shown) - MAX_SETTINGS}" if len(shown) > MAX_SETTINGS else ""
    return ", ".join(shown[:MAX_SETTINGS]) + more


def _children(obj, found: list, seen: set, depth: int = 0) -> None:
    """Collect the estimators directly below this object."""
    if id(obj) in seen or depth > 50:
        return
    seen.add(id(obj))
    if _is_estimator(obj):
        found.append(obj)
    elif isinstance(obj, Stub):
        for part in (obj.args, obj.state, obj.items):
            _children(part, found, seen, depth + 1)
    elif isinstance(obj, dict):
        for value in obj.values():
            _children(value, found, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _children(item, found, seen, depth + 1)
    elif type(obj).__name__ == "ndarray" and obj.dtype == object:
        for item in obj.ravel().tolist():
            _children(item, found, seen, depth + 1)


def outline(scan: Scan) -> list[str]:
    """One line for each part of the model. Equal parts in a row, such as the trees of a forest, become one line."""
    lines: list[str] = []

    def add(stub: Stub, depth: int, count: int = 1) -> None:
        name = stub.path[1]
        prefix = "  " * depth + (f"{count} x " if count > 1 else "")
        settings = _settings(stub)
        lines.append(f"{prefix}{name}" + (f"  ({settings})" if settings and depth < 2 else ""))
        if depth >= MAX_DEPTH:
            return
        found: list = []
        _children(stub.state, found, {id(stub)})
        groups: list[list] = []
        for child in found:
            if groups and groups[-1][0].path == child.path:
                groups[-1].append(child)
            else:
                groups.append([child])
        for group in groups:
            # An ensemble also keeps one unfitted copy of its part as a pattern. Do not count it.
            fitted = [child for child in group if any(key.endswith("_") for key in child.state)]
            add((fitted or group)[0], depth + 1, len(fitted or group))

    roots: list = []
    _children(scan.root, roots, set())
    for root in roots[:1]:
        add(root, 0)
    return lines


@dataclass
class Description:
    file: str
    size: int = 0
    kind: str = "pickle"
    saved: list[str] = field(default_factory=list)
    outline: list[str] = field(default_factory=list)
    inputs: object = None           # a Contract, when the model can be loaded
    classes: list | None = None
    has_card: bool = False
    note: str = ""

    def __str__(self) -> str:
        size = f"{self.size / 1024:.0f} KB" if self.size < 1024 * 1024 else f"{self.size / 1024 / 1024:.1f} MB"
        lines = [self.file, f"  File         {self.kind}, {size}, saved with scikit-learn {', '.join(self.saved) or 'unknown'}",
                 f"  Proof card   {'yes' if self.has_card else 'no'}", "  Parts"]
        lines += ["    " + line for line in self.outline] or ["    no scikit-learn model found"]
        if self.inputs is not None:
            lines.append("  Input")
            lines += ["    " + line for line in str(self.inputs).splitlines()]
        if self.classes:
            shown = ", ".join(str(item) for item in self.classes[:10]) + (", ..." if len(self.classes) > 10 else "")
            lines.append(f"  Classes      {shown}")
        if self.note:
            lines.append(f"  Note         {self.note}")
        return "\n".join(lines)


def describe_scan(file: Path, scan: Scan) -> Description:
    from ._card import card_path

    return Description(file=str(file), size=file.stat().st_size, kind=scan.kind, saved=scan.versions,
                       outline=outline(scan), has_card=card_path(file).exists())
