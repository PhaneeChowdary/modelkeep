"""Read a model file and run no code from it.

Python can run any code when it loads a pickle file. The scan loads the file with stand-in classes,
so no code from the file runs. The scan finds the scikit-learn version that saved each part.
It also finds the names of all classes in the file.
"""
from __future__ import annotations

import bz2
import gzip
import inspect
import io
import lzma
import pickle
import zlib
from dataclasses import dataclass, field

# In a scan, only these names from a file become real objects. They make NumPy arrays and do nothing else.
ARRAY_NAMES = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}
SAFE_MODULES = ("sklearn", "numpy", "scipy", "joblib", "collections", "copyreg", "builtins", "_loss", "_codecs")
PACKERS = ((b"\x1f\x8b", gzip.decompress), (b"BZh", bz2.decompress), (b"\xfd7zXZ", lzma.decompress),
           (b"\x5d\x00", lzma.decompress), (b"\x78", zlib.decompress))


class Stub:
    """Takes the place of an object in the file. It keeps the data that the file gives it and does nothing else."""

    path = ("", "")
    args: tuple = ()
    state = None
    items: tuple = ()

    # Pickle can make an object and not call __init__. Thus each method must work on an empty object.
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def _keep(self, values):
        self.items = tuple(self.items) + tuple(values)

    def append(self, item):
        self._keep([item])

    def extend(self, items):
        self._keep(items)

    add = append

    def __setitem__(self, key, value):
        self._keep([(key, value)])

    def __call__(self, *args, **kwargs):
        return Stub(*args)


@dataclass
class Scan:
    kind: str = "pickle"
    packed: bool = False
    estimators: list[tuple[str, str]] = field(default_factory=list)  # (class path, version); the outer part is first
    names: set[tuple[str, str]] = field(default_factory=set)
    params: dict[int, dict] = field(default_factory=dict)
    root: object = None

    @property
    def versions(self) -> list[str]:
        return sorted({version for _, version in self.estimators if version}, key=version_key)

    @property
    def unknown_names(self) -> list[str]:
        return sorted(f"{module}.{name}" for module, name in self.names
                      if module.split(".")[0] not in SAFE_MODULES)


def version_key(text: str) -> tuple:
    parts = []
    for part in text.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def unpack(data: bytes) -> tuple[bytes, bool]:
    """Return the plain bytes of a file. joblib can compress a file."""
    if data[:1] == b"\x80":
        return data, False
    for magic, expand in PACKERS:
        if data.startswith(magic):
            try:
                return expand(data), True
            except (OSError, EOFError, zlib.error, lzma.LZMAError):
                continue
    return data, False


def _stub_class(module: str, name: str, cache: dict) -> type:
    key = (module, name)
    if key not in cache:
        cache[key] = type(name, (Stub,), {"path": key})
    return cache[key]


def _finder(scan: Scan, cache: dict, real):
    def find_class(self, module, name):
        scan.names.add((module, name))
        if (module, name) in ARRAY_NAMES and real is not None:
            return real(self, module, name)
        return _stub_class(module, name, cache)
    return find_class


def _collect(obj, scan: Scan, seen: set, depth: int = 0) -> None:
    if id(obj) in seen or depth > 200:
        return
    seen.add(id(obj))
    if isinstance(obj, Stub):
        state = obj.state
        if isinstance(state, dict) and "_sklearn_version" in state:
            scan.estimators.append((".".join(obj.path), str(state["_sklearn_version"])))
            scan.params[len(scan.estimators) - 1] = {
                key: value for key, value in state.items() if isinstance(value, (str, int, float, bool, type(None)))}
        for part in (obj.args, state, obj.items):
            _collect(part, scan, seen, depth + 1)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect(key, scan, seen, depth + 1)
            _collect(value, scan, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _collect(item, scan, seen, depth + 1)
    elif type(obj).__name__ == "ndarray" and getattr(obj, "dtype", None) is not None and obj.dtype == object:
        for item in obj.ravel().tolist():
            _collect(item, scan, seen, depth + 1)


def scan_bytes(data: bytes) -> Scan:
    data, packed = unpack(data)
    scan = Scan(packed=packed)
    cache: dict = {}
    is_joblib = b"NumpyArrayWrapper" in data
    if is_joblib:
        scan.kind = "joblib"
        try:
            import joblib.numpy_pickle as npk
        except ImportError as error:
            raise RuntimeError("This file was saved with joblib. Install joblib and numpy to read it.") from error
        reader = type("Reader", (npk.NumpyUnpickler,),
                      {"find_class": _finder(scan, cache, npk.NumpyUnpickler.find_class)})
        # New joblib versions have one more argument.
        wanted = inspect.signature(npk.NumpyUnpickler.__init__).parameters
        extra = {"ensure_native_byte_order": False} if "ensure_native_byte_order" in wanted else {}
        root = reader("", io.BytesIO(data), mmap_mode=None, **extra).load()
    else:
        reader = type("Reader", (pickle.Unpickler,), {"find_class": _finder(scan, cache, None)})
        root = reader(io.BytesIO(data)).load()
    scan.root = root
    _collect(root, scan, set())
    return scan
