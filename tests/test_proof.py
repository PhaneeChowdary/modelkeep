"""Save a model with proof, then check the proof."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import modelkeep
from modelkeep.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
PEOPLE = pd.DataFrame({"age": [30, 45, 22, 60, 41, 35], "city": ["NY", "LA", "NY", "SF", "LA", "SF"],
                       "income": [50.0, 80.0, 30.0, 90.0, 75.0, 52.0]})
LABELS = [0, 1, 0, 1, 1, 0]


def table_model():
    columns = ColumnTransformer([("numbers", StandardScaler(), ["age", "income"]), ("places", OneHotEncoder(), ["city"])])
    return Pipeline([("columns", columns), ("forest", RandomForestClassifier(n_estimators=5, random_state=0))]).fit(PEOPLE, LABELS)


@pytest.mark.parametrize("name", ["model.pkl", "model.joblib", "model.joblib.gz"])
def test_save_then_verify(tmp_path, name):
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(np.arange(40).reshape(10, 4), [0, 1] * 5)
    card = modelkeep.save(model, tmp_path / name, np.arange(40).reshape(10, 4))
    assert card.name == name + ".card.json"
    verdict = modelkeep.verify(tmp_path / name)
    assert verdict.ok is True
    assert {check.call for check in verdict.checks} == {"predict", "predict_proba"}


def test_table_and_text_samples(tmp_path):
    modelkeep.save(table_model(), tmp_path / "table.joblib", PEOPLE)
    assert modelkeep.verify(tmp_path / "table.joblib").ok is True
    texts = ["free prize now", "meeting at noon", "win money free", "lunch tomorrow"]
    words = Pipeline([("words", TfidfVectorizer()), ("model", LogisticRegression())]).fit(texts, [1, 0, 1, 0])
    modelkeep.save(words, tmp_path / "words.pkl", texts)
    assert modelkeep.verify(tmp_path / "words.pkl").ok is True


def test_verify_finds_a_changed_model(tmp_path):
    X = np.arange(40).reshape(10, 4)
    modelkeep.save(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, [0, 1] * 5), tmp_path / "model.pkl", X)
    # Put a different model in the file. The card still holds the old results.
    other = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, [1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    (tmp_path / "model.pkl").write_bytes(__import__("pickle").dumps(other))
    verdict = modelkeep.verify(tmp_path / "model.pkl")
    assert verdict.ok is False
    assert "DIFFERENT" in str(verdict)
    assert main(["verify", str(tmp_path / "model.pkl")]) == 2


def test_verify_without_a_card(tmp_path):
    shutil.copy(FIXTURES / "1.8.0" / "logistic.pkl", tmp_path / "model.pkl")
    verdict = modelkeep.verify(tmp_path / "model.pkl")
    assert verdict.ok is None
    assert main(["verify", str(tmp_path / "model.pkl")]) == 1


def test_save_does_not_overwrite(tmp_path):
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    modelkeep.save(model, tmp_path / "model.pkl")
    with pytest.raises(FileExistsError):
        modelkeep.save(model, tmp_path / "model.pkl")
    modelkeep.save(model, tmp_path / "model.pkl", [[0.5]], overwrite=True)
    assert modelkeep.verify(tmp_path / "model.pkl").ok is True


def old_file_with_card(tmp_path, version="1.2.2", name="forest_clf"):
    """A file from an old version, with the card that modelkeep would have written at that time."""
    expected = json.loads((FIXTURES / version / "expected.json").read_text())
    entry = expected["models"][name]
    shutil.copy(FIXTURES / version / (name + ".pkl"), tmp_path / "old.pkl")
    card = {"modelkeep": 1, "versions": {"scikit-learn": version, "python": expected["python"]},
            "proof": {"sample": {"kind": "array", "rows": expected["X_new"]}, "results": {entry["call"]: entry["result"]}}}
    (tmp_path / "old.pkl.card.json").write_text(json.dumps(card))
    return tmp_path / "old.pkl"


def test_verify_an_old_file_after_an_upgrade(tmp_path):
    verdict = modelkeep.verify(old_file_with_card(tmp_path))
    assert verdict.ok is True
    assert verdict.repairs > 0
    assert verdict.then["scikit-learn"] == "1.2.2"


def test_fix_checks_the_proof_and_writes_a_new_card(tmp_path):
    report = modelkeep.fix(old_file_with_card(tmp_path))
    assert "gives the saved results" in report.proof
    assert (tmp_path / "old.fixed.pkl.card.json").exists()
    verdict = modelkeep.verify(tmp_path / "old.fixed.pkl")
    assert verdict.ok is True and verdict.repairs == 0


def test_fix_stops_when_the_proof_fails(tmp_path):
    old = old_file_with_card(tmp_path)
    card = json.loads((tmp_path / "old.pkl.card.json").read_text())
    card["proof"]["results"]["predict_proba"][0] = [0.123, 0.877]
    (tmp_path / "old.pkl.card.json").write_text(json.dumps(card))
    with pytest.raises(modelkeep.CannotRepair, match="saved results"):
        modelkeep.fix(old)
    assert not (tmp_path / "old.fixed.pkl").exists()
