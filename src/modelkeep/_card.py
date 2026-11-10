"""Save a model with proof of its results, and check the proof later.

The proof is a small JSON file next to the model. It holds some input rows and the results that the model
gave for them on the day of the save. After an upgrade, the same rows must give the same results.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

CARD_SUFFIX = ".card.json"
CALLS = ("predict", "predict_proba", "decision_function", "transform", "score_samples")
JOBLIB_ENDINGS = (".joblib", ".gz", ".bz2", ".xz", ".lzma", ".z")
MAX_ROWS = 20
PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "pandas")


def card_path(file) -> Path:
    return Path(str(file) + CARD_SUFFIX)


def versions() -> dict:
    found = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def _is_table(sample) -> bool:
    return hasattr(sample, "columns") and hasattr(sample, "iloc")


def pack_sample(sample) -> dict:
    """Change the sample rows to plain JSON data. Keep at most MAX_ROWS rows."""
    import numpy as np

    if _is_table(sample):
        frame = sample.iloc[:MAX_ROWS]
        rows = [[None if value != value else value for value in row] for row in frame.astype(object).values.tolist()]
        return {"kind": "table", "columns": [str(name) for name in frame.columns],
                "types": [str(kind) for kind in frame.dtypes], "rows": rows}
    if isinstance(sample, (list, tuple)) and sample and all(isinstance(item, str) for item in sample):
        return {"kind": "text", "rows": list(sample[:MAX_ROWS])}
    return {"kind": "array", "rows": np.asarray(sample)[:MAX_ROWS].tolist()}


def unpack_sample(packed: dict):
    import numpy as np

    if packed["kind"] == "text":
        return list(packed["rows"])
    if packed["kind"] == "array":
        return np.array(packed["rows"])
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("The proof uses a pandas table. Install pandas to check it.") from error
    frame = pd.DataFrame(packed["rows"], columns=packed["columns"])
    for name, kind in zip(packed["columns"], packed["types"], strict=True):
        try:
            frame[name] = frame[name].astype(kind)
        except (TypeError, ValueError):
            pass
    return frame


def results(model, data) -> dict:
    """Call each result method that the model has. A method that fails is left out."""
    import numpy as np

    found = {}
    for call in CALLS:
        if not hasattr(model, call):
            continue
        try:
            value = getattr(model, call)(data)
        except Exception:  # noqa: BLE001 - some models have a method that does not apply to them
            continue
        if hasattr(value, "toarray"):
            value = value.toarray()
        value = np.asarray(value)
        found[call] = value.tolist() if value.dtype.kind in "fiub" else [str(item) for item in value.ravel()]
    return found


def write_model(model, file: Path, compress: bool = False) -> None:
    if file.name.lower().endswith(JOBLIB_ENDINGS):
        import joblib

        joblib.dump(model, file, compress=3 if compress or not file.name.lower().endswith(".joblib") else 0)
    else:
        file.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def make_card(model, file: Path, proof: dict | None) -> dict:
    from ._contract import expects

    try:
        inputs = expects(model).as_dict()
    except Exception:  # noqa: BLE001 - the card is still useful without the input list
        inputs = None
    classes = getattr(model, "classes_", None)
    return {
        "modelkeep": 1,
        "file": file.name,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": versions(),
        "model": type(model).__name__,
        "inputs": inputs,
        "classes": [str(item) for item in classes] if classes is not None and not isinstance(classes, list) else None,
        "proof": proof,
    }


def save(model, file, sample=None, *, overwrite: bool = False) -> Path:
    """Save the model. With sample rows, also save proof of its results. Return the path of the card."""
    file = Path(file)
    card = card_path(file)
    if (file.exists() or card.exists()) and not overwrite:
        raise FileExistsError(f"{file} or its card exists. Give another name, or use overwrite=True.")
    proof = None
    if sample is not None:
        packed = pack_sample(sample)
        proof = {"sample": packed, "results": results(model, unpack_sample(packed))}
        if not proof["results"]:
            raise ValueError("The model gave no result for the sample rows. Give rows that the model accepts.")
    write_model(model, file)
    card.write_text(json.dumps(make_card(model, file, proof), indent=1))
    return card


@dataclass
class Check:
    call: str
    same: bool
    difference: float | None = None


@dataclass
class Verdict:
    file: str
    ok: bool | None = None          # None: there is no proof to check
    reason: str = ""
    checks: list[Check] = field(default_factory=list)
    then: dict = field(default_factory=dict)
    now: dict = field(default_factory=dict)
    repairs: int = 0

    def __str__(self) -> str:
        result = {True: "The model gives the saved results.", False: "The model gives DIFFERENT results.",
                  None: "No proof to check."}[self.ok]
        lines = [self.file, f"  Result       {result}"]
        if self.reason:
            lines.append(f"  Reason       {self.reason}")
        if self.then:
            lines.append(f"  Saved with   scikit-learn {self.then.get('scikit-learn', '?')}, Python {self.then.get('python', '?')}")
            lines.append(f"  Now          scikit-learn {self.now.get('scikit-learn', '?')}, Python {self.now.get('python', '?')}")
        if self.repairs:
            lines.append(f"  Repairs      {self.repairs} repairs were necessary to load the file")
        for check in self.checks:
            detail = "same" if check.same else "different"
            if check.difference is not None and not check.same:
                detail += f", largest difference {check.difference:.3g}"
            lines.append(f"    {check.call:<18} {detail}")
        return "\n".join(lines)


def compare(model, proof: dict, tolerance: float) -> list[Check]:
    import numpy as np

    now = results(model, unpack_sample(proof["sample"]))
    checks = []
    for call, old in proof["results"].items():
        if call not in now:
            checks.append(Check(call, False))
            continue
        old_values, new_values = np.asarray(old), np.asarray(now[call])
        if old_values.dtype.kind in "fiub" and new_values.dtype.kind in "fiub":
            if old_values.shape != new_values.shape:
                checks.append(Check(call, False))
                continue
            same = bool(np.allclose(new_values, old_values, atol=tolerance, rtol=0, equal_nan=True))
            gap = np.abs(new_values.astype(float) - old_values.astype(float))
            checks.append(Check(call, same, float(np.nanmax(gap)) if gap.size else 0.0))
        else:
            checks.append(Check(call, old_values.tolist() == new_values.tolist()))
    return checks
