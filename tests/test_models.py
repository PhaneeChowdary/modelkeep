"""Each file in tests/fixtures was saved by an old scikit-learn version, with the results that version gave."""
import json
from pathlib import Path

import numpy as np
import pytest

import modelkeep

FIXTURES = Path(__file__).parent / "fixtures"
VERSIONS = sorted((folder.name for folder in FIXTURES.iterdir() if folder.is_dir()),
                  key=lambda name: tuple(map(int, name.split("."))))
# AdaBoost used the SAMME.R method by default until scikit-learn 1.6 removed it.
REMOVED_METHOD = {"1.0.2", "1.1.3", "1.2.2", "1.3.2", "1.4.2", "1.5.2"}


def cases():
    for version in VERSIONS:
        expected = json.loads((FIXTURES / version / "expected.json").read_text())
        for name in expected["models"]:
            names = ["compressed.joblib.gz"] if name == "compressed" else [name + ".pkl", name + ".joblib"]
            for file in names:
                yield pytest.param(version, name, file, id=f"{version}-{file}")


@pytest.mark.parametrize("version, name, file", cases())
def test_old_model_gives_old_results(version, name, file):
    expected = json.loads((FIXTURES / version / "expected.json").read_text())
    entry = expected["models"][name]
    path = FIXTURES / version / file
    if name == "adaboost" and version in REMOVED_METHOD:
        with pytest.raises(modelkeep.CannotRepair, match="SAMME.R"):
            modelkeep.load(path)
        return
    model = modelkeep.load(path)
    data = expected["text_new"] if entry.get("text") else np.array(expected["X_new"])
    assert np.allclose(getattr(model, entry["call"])(data), entry["result"], atol=1e-8)


@pytest.mark.parametrize("version", VERSIONS)
def test_check_reads_the_saved_version(version):
    report = modelkeep.check(FIXTURES / version / "forest_clf.pkl")
    assert report.saved == [version]
    assert report.parts[:2] == ["RandomForestClassifier", "DecisionTreeClassifier"]
    assert report.status == ("repairable" if version < "1.4" else "current")


def test_check_finds_the_silent_change():
    # A tree classifier from 1.3 loads without an error, but predict_proba returns counts.
    report = modelkeep.check(FIXTURES / "1.3.2" / "tree_clf.pkl")
    assert report.count("broken") == 0
    assert report.count("silent") == 1
    assert "wrong results" in str(report)


def test_check_reports_a_refused_model():
    report = modelkeep.check(FIXTURES / "1.5.2" / "adaboost.pkl")
    assert report.status == "refused"
    assert "SAMME.R" in report.error


def test_allow_changed_loads_a_refused_model():
    model = modelkeep.load(FIXTURES / "1.5.2" / "adaboost.pkl", allow_changed=True)
    assert model.predict(np.zeros((1, 4))).shape == (1,)
