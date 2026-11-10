import json
import pickle
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

import modelkeep
from modelkeep.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def expected(version, name):
    data = json.loads((FIXTURES / version / "expected.json").read_text())
    return np.array(data["X_new"]), data["models"][name]


@pytest.mark.parametrize("file, reader", [("forest_clf.pkl", lambda path: pickle.loads(path.read_bytes())),
                                         ("forest_clf.joblib", joblib.load)])
def test_fixed_file_loads_plainly_with_old_results(tmp_path, file, reader):
    target = tmp_path / file
    report = modelkeep.fix(FIXTURES / "1.2.2" / file, target)
    assert report.status == "fixed"
    X, entry = expected("1.2.2", "forest_clf")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = reader(target)
        assert np.allclose(model.predict_proba(X), entry["result"])
    assert modelkeep.check(target).status == "current"


def test_fix_keeps_the_old_file_and_does_not_overwrite(tmp_path):
    source = tmp_path / "model.pkl"
    source.write_bytes((FIXTURES / "1.2.2" / "tree_reg.pkl").read_bytes())
    before = source.read_bytes()
    modelkeep.fix(source)
    assert source.read_bytes() == before
    assert (tmp_path / "model.fixed.pkl").exists()
    with pytest.raises(FileExistsError):
        modelkeep.fix(source)
    modelkeep.fix(source, overwrite=True)


def test_command_line_exit_codes(tmp_path, capsys):
    assert main(["check", str(FIXTURES / "1.8.0" / "logistic.pkl")]) == 0
    assert main(["check", str(FIXTURES / "1.2.2" / "forest_clf.pkl")]) == 1
    assert main(["check", str(FIXTURES / "1.5.2" / "adaboost.pkl")]) == 2
    assert main(["fix", str(FIXTURES / "1.2.2" / "forest_clf.pkl"), "-o", str(tmp_path / "new.pkl")]) == 0
    assert main(["fix", str(FIXTURES / "1.5.2" / "adaboost.pkl"), "-o", str(tmp_path / "no.pkl")]) == 2
    assert "modelkeep fix" in capsys.readouterr().out
