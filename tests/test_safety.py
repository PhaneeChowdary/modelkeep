import fractions
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import modelkeep
from modelkeep._scan import scan_bytes

FIXTURES = Path(__file__).parent / "fixtures"


class RunsACommand:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def test_scan_does_not_run_code(tmp_path):
    marker = tmp_path / "ran"
    scan = scan_bytes(pickle.dumps({"model": RunsACommand(marker)}))
    assert not marker.exists()
    assert scan.unknown_names == ["posix.system"] or scan.unknown_names == ["nt.system"]


def test_check_and_load_refuse_unknown_code(tmp_path):
    marker = tmp_path / "ran"
    file = tmp_path / "model.pkl"
    file.write_bytes(pickle.dumps(RunsACommand(marker)))
    assert modelkeep.check(file).status == "unsafe"
    with pytest.raises(modelkeep.UnsafeFile):
        modelkeep.load(file)
    assert not marker.exists()


def test_trust_loads_other_classes(tmp_path):
    file = tmp_path / "value.pkl"
    file.write_bytes(pickle.dumps({"ratio": fractions.Fraction(1, 3)}))
    with pytest.raises(modelkeep.UnsafeFile):
        modelkeep.load(file)
    assert modelkeep.load(file, trust=True) == {"ratio": fractions.Fraction(1, 3)}


def test_scan_needs_no_numpy_for_a_pickle_file():
    code = ("import sys; from modelkeep._scan import scan_bytes; from pathlib import Path;"
            f"scan = scan_bytes(Path({str(FIXTURES / '1.0.2' / 'forest_clf.pkl')!r}).read_bytes());"
            "assert scan.versions == ['1.0.2'], scan.versions;"
            "assert 'numpy' not in sys.modules and 'sklearn' not in sys.modules")
    source = str(Path(modelkeep.__file__).parents[1])
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {source!r}); {code}"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_damaged_file_gives_a_report(tmp_path):
    file = tmp_path / "bad.pkl"
    file.write_bytes(b"this is not a model")
    assert modelkeep.check(file).status == "failed"
