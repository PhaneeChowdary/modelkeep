"""modelkeep keeps scikit-learn model files in a usable condition.

    import modelkeep

    modelkeep.save(model, "model.joblib", X[:20])   # save the model with proof of its results
    modelkeep.verify("model.joblib")                # after an upgrade: does it give the same results?
    modelkeep.check("old.pkl")                      # what is in the file, and does it load?
    model = modelkeep.load("old.pkl")               # load a file that an old scikit-learn saved
    modelkeep.fix("old.pkl")                        # write old.fixed.pkl for the installed version
    modelkeep.describe("model.joblib")              # parts, settings and expected input
    modelkeep.validate(model, rows)                 # problems in the input, as sentences
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

from . import _card
from ._card import CARD_SUFFIX, Check, Verdict, card_path, save
from ._contract import Column, Contract, expects, validate
from ._describe import Description, describe_scan
from ._load import BROKEN, FILLED, LOST, SILENT, CannotRepair, Lost, Note, UnsafeFile, _objects, repair_bytes
from ._scan import Scan, scan_bytes, version_key

__version__ = "0.2.0"
__all__ = ["CannotRepair", "Check", "Column", "Contract", "Description", "Note", "Report", "UnsafeFile", "Verdict",
           "check", "describe", "expects", "find_models", "fix", "load", "repair", "save", "validate", "verify"]

CURRENT, REPAIRABLE, FIXED, REFUSED, UNSAFE, FAILED, UNKNOWN = (
    "current", "repairable", "fixed", "refused", "unsafe", "failed", "unknown")
LABELS = {BROKEN: "stops a plain load", SILENT: "wrong results", LOST: "removed part", FILLED: "default value"}
RESULTS = {
    CURRENT: "The file loads with the installed version. No repair is necessary.",
    REPAIRABLE: "A plain load fails or gives wrong results. modelkeep can repair this file.",
    FIXED: "Repaired and saved for the installed version.",
    REFUSED: "modelkeep cannot give the old results for this model.",
    UNSAFE: "The file names code that is not from scikit-learn. modelkeep did not load it.",
    FAILED: "modelkeep could not load this file.",
    UNKNOWN: "scikit-learn is not installed here, so modelkeep only read the file.",
}
MODEL_ENDINGS = (".pkl", ".pickle", ".joblib", ".sav", ".model", ".pkl.gz", ".joblib.gz", ".joblib.bz2", ".joblib.xz")
# The first and the last Python version that each scikit-learn release has a build for. Source: PyPI.
PYTHON_RANGE = {
    (0, 24): ("3.6", "3.9"), (1, 0): ("3.7", "3.10"), (1, 1): ("3.8", "3.11"), (1, 2): ("3.8", "3.11"),
    (1, 3): ("3.8", "3.12"), (1, 4): ("3.9", "3.12"), (1, 5): ("3.9", "3.13"), (1, 6): ("3.9", "3.13"),
    (1, 7): ("3.10", "3.14"), (1, 8): ("3.11", "3.14"), (1, 9): ("3.11", "3.15"),
}


@dataclass
class Report:
    file: str
    kind: str = "pickle"
    packed: bool = False
    saved: list[str] = field(default_factory=list)
    installed: str | None = None
    parts: list[str] = field(default_factory=list)
    notes: list[tuple[Note, int]] = field(default_factory=list)
    status: str = UNKNOWN
    error: str = ""
    proof: str = ""

    def count(self, kind: str) -> int:
        return sum(number for note, number in self.notes if note.kind == kind)

    @property
    def old_way(self) -> str:
        """The setup that loads the file without modelkeep."""
        if not self.saved:
            return ""
        first, last = PYTHON_RANGE.get(version_key(self.saved[0])[:2], ("", ""))
        return f"scikit-learn {self.saved[0]}" + (f" on Python {first} to {last}" if first else "")

    def as_dict(self) -> dict:
        data = asdict(self)
        data["notes"] = [{"kind": note.kind, "part": note.part, "text": note.text, "count": number}
                         for note, number in self.notes]
        data["old_way"] = self.old_way
        return data

    def __str__(self) -> str:
        saved = ", ".join(self.saved) or "not found in the file"
        model = " > ".join(filter(None, [", ".join(self.parts[:1]), ", ".join(self.parts[1:4])])) or "unknown"
        lines = [
            self.file,
            f"  Saved with   scikit-learn {saved} ({self.kind}{', compressed' if self.packed else ''})",
            f"  Installed    scikit-learn {self.installed or 'not installed'}",
            f"  Model        {model}",
            f"  Result       {RESULTS[self.status]}",
        ]
        if self.error:
            lines.append(f"  Reason       {self.error}")
        shown = [(note, number) for note, number in self.notes if note.kind != FILLED]
        if shown:
            lines.append("  Repairs")
            for note, number in shown:
                lines.append(f"    {LABELS[note.kind]:<19} {number} x {note.part}: {note.text}")
        filled = self.count(FILLED)
        if filled:
            lines.append(f"    {LABELS[FILLED]:<19} {filled} settings that the old version did not have")
        if self.proof:
            lines.append(f"  Proof        {self.proof}")
        if self.status in (REPAIRABLE, REFUSED, FAILED) and self.old_way:
            lines.append(f"  Other way    install {self.old_way}")
        return "\n".join(lines)


def _installed() -> str | None:
    try:
        return metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return None


def _parts(scan: Scan) -> list[str]:
    names = []
    for path, _ in scan.estimators:
        name = path.split(".")[-1]
        if name not in names:
            names.append(name)
    return names


def _report(file, scan: Scan) -> Report:
    return Report(file=str(file), kind=scan.kind, packed=scan.packed, saved=scan.versions,
                  installed=_installed(), parts=_parts(scan))


def _group(notes: list[Note]) -> list[tuple[Note, int]]:
    order = {BROKEN: 0, SILENT: 1, LOST: 2, FILLED: 3}
    return sorted(Counter(notes).items(), key=lambda item: (order[item[0].kind], item[0].part, item[0].text))


def _status(report: Report) -> str:
    needs_repair = report.count(BROKEN) or report.count(SILENT) or report.count(LOST)
    return REPAIRABLE if needs_repair else CURRENT


def repair(file, *, trust: bool = False, allow_changed: bool = False):
    """Load the model and repair it. Return (model, report)."""
    data = Path(file).read_bytes()
    scan = scan_bytes(data)
    report = _report(file, scan)
    model, notes, _ = repair_bytes(data, trust=trust, allow_changed=allow_changed, scan=scan)
    report.notes = _group(notes)
    report.status = _status(report)
    return model, report


def load(file, *, trust: bool = False, allow_changed: bool = False):
    """Load the model and repair it. Return the model."""
    return repair(file, trust=trust, allow_changed=allow_changed)[0]


def check(file, *, trust: bool = False) -> Report:
    """Say what is in the file and what a load needs. A file that names unknown code is read but not loaded."""
    data = Path(file).read_bytes()
    try:
        scan = scan_bytes(data)
    except Exception as error:  # noqa: BLE001 - a damaged file can fail in many ways
        return Report(file=str(file), installed=_installed(), status=FAILED, error=f"{type(error).__name__}: {error}")
    report = _report(file, scan)
    if scan.unknown_names and not trust:
        report.status, report.error = UNSAFE, ", ".join(scan.unknown_names[:5])
        return report
    if report.installed is None:
        return report
    try:
        _, notes, _ = repair_bytes(data, trust=trust, scan=scan)
    except CannotRepair as error:
        report.status, report.error = REFUSED, str(error)
    except Exception as error:  # noqa: BLE001 - report the cause, do not stop the caller
        report.status, report.error = FAILED, f"{type(error).__name__}: {error}"
    else:
        report.notes = _group(notes)
        report.status = _status(report)
    return report


def find_models(folder) -> list[Path]:
    """All model files below a folder, found by the end of the file name."""
    return sorted(path for path in Path(folder).rglob("*")
                  if path.is_file() and path.name.lower().endswith(MODEL_ENDINGS))


def _read_card(file) -> dict | None:
    path = card_path(file)
    if not path.exists():
        return None
    try:
        card = json.loads(path.read_text())
    except ValueError:
        return None
    return card if isinstance(card, dict) else None


def verify(file, *, tolerance: float = 1e-8, trust: bool = False) -> Verdict:
    """Load the model and compare its results with the proof that modelkeep.save wrote."""
    verdict = Verdict(file=str(file), now=_card.versions())
    card = _read_card(file)
    if card is None or not card.get("proof"):
        verdict.reason = f"No proof for this file. Save the model with modelkeep.save(model, file, sample) to get {CARD_SUFFIX}."
        return verdict
    verdict.then = card.get("versions", {})
    try:
        model, report = repair(file, trust=trust)
    except (CannotRepair, UnsafeFile) as error:
        verdict.ok, verdict.reason = False, str(error)
        return verdict
    verdict.repairs = report.count(BROKEN) + report.count(SILENT) + report.count(LOST)
    verdict.checks = _card.compare(model, card["proof"], tolerance)
    verdict.ok = all(item.same for item in verdict.checks)
    return verdict


def fix(file, target=None, *, trust: bool = False, overwrite: bool = False) -> Report:
    """Repair the model and save it for the installed scikit-learn version. The old file stays.

    If the old file has a proof card, the new file must give the saved results, and it gets its own card.
    """
    source = Path(file)
    if target is None:
        stem = source.name.split(".")[0]
        target = source.with_name(stem + ".fixed" + source.name[len(stem):])
    target = Path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists. Give another name, or use overwrite=True.")
    model, report = repair(source, trust=trust)
    lost = sorted({type(obj).__name__ for obj in _objects(model) if isinstance(obj, Lost)})
    if lost:
        hint = " Save its best_estimator_ instead." if hasattr(model, "best_estimator_") else ""
        raise CannotRepair(f"The model holds parts that no longer exist ({', '.join(lost)}), so the saved file "
                           f"would need modelkeep to load.{hint}")
    card = _read_card(source)
    if card and card.get("proof"):
        checks = _card.compare(model, card["proof"], 1e-8)
        if not all(item.same for item in checks):
            different = ", ".join(item.call for item in checks if not item.same)
            raise CannotRepair(f"The repaired model does not give the saved results for: {different}. No file was written.")
        report.proof = f"the repaired model gives the saved results ({len(checks)} result {'check' if len(checks) == 1 else 'checks'})"
    _card.write_model(model, target, compress=report.packed)
    saved = scan_bytes(target.read_bytes())
    if saved.versions != [report.installed]:
        raise RuntimeError(f"The new file reports versions {saved.versions}, expected {report.installed}.")
    if card and card.get("proof"):
        card_path(target).write_text(json.dumps(_card.make_card(model, target, card["proof"]), indent=1))
    report.file, report.status = f"{source} -> {target}", FIXED
    return report


def describe(file, *, trust: bool = False) -> Description:
    """Describe the parts of a model file. If the file is safe to load, also give the input that it expects."""
    file = Path(file)
    scan = scan_bytes(file.read_bytes())
    description = describe_scan(file, scan)
    if scan.unknown_names and not trust:
        description.note = "The file names unknown code, so modelkeep did not load it: " + ", ".join(scan.unknown_names[:3])
        return description
    if _installed() is None:
        description.note = "scikit-learn is not installed here, so the input is unknown."
        return description
    try:
        model = load(file, trust=trust, allow_changed=True)
        description.inputs = expects(model)
        classes = getattr(model, "classes_", None)
        description.classes = [item.item() if hasattr(item, "item") else item for item in classes] if classes is not None else None
    except Exception as error:  # noqa: BLE001 - the list of parts is still useful
        description.note = f"The input is unknown: {type(error).__name__}: {error}"
    return description
