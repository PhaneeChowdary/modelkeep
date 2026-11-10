"""Load each test model with modelkeep and compare its results with the results of the old version."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelkeep._load import CannotRepair, repair_bytes  # noqa: E402

warnings.simplefilter("ignore")
root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
total = failed = refused = 0
for folder in sorted(root.iterdir(), key=lambda p: tuple(map(int, p.name.split(".")))):
    expected = json.loads((folder / "expected.json").read_text())
    bad = {}
    for name, entry in expected["models"].items():
        files = [folder / "compressed.joblib.gz"] if name == "compressed" else [folder / (name + ".pkl"), folder / (name + ".joblib")]
        for file in files:
            total += 1
            try:
                model, notes, _ = repair_bytes(file.read_bytes())
                X = expected["text_new"] if entry.get("text") else np.array(expected["X_new"])
                got = getattr(model, entry["call"])(X)
                if not np.allclose(got, entry["result"], atol=1e-8):
                    bad[file.name] = "DIFFERENT"
            except CannotRepair:
                refused += 1
            except Exception as error:  # noqa: BLE001
                bad[file.name] = type(error).__name__ + ": " + str(error)[:110]
    failed += len(bad)
    print(folder.name, "ok" if not bad else "")
    for key, value in bad.items():
        print("   ", key, "->", value)
print(f"{total - failed - refused}/{total} files give the old results, {refused} refused, {failed} failed")
