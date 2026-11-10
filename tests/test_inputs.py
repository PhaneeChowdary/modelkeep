"""The input that a model expects, and the check of rows against it."""
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from test_proof import PEOPLE, table_model

import modelkeep
from modelkeep.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_table_model_names_kinds_and_values():
    contract = modelkeep.expects(table_model())
    assert [(column.name, column.kind) for column in contract.columns] == [
        ("age", "number"), ("city", "category"), ("income", "number")]
    assert contract.columns[1].allowed == ["LA", "NY", "SF"]
    assert contract.json_schema()["properties"]["city"] == {"enum": ["LA", "NY", "SF"]}
    assert contract.json_schema()["required"] == ["age", "city", "income"]


def test_validate_gives_one_sentence_for_each_problem():
    model = table_model()
    assert modelkeep.validate(model, PEOPLE) == []
    assert modelkeep.validate(model, {"age": 30, "city": "NY", "income": 10}) == []
    problems = modelkeep.validate(model, [{"age": "abc", "city": "Paris", "income": 5}, {"age": 3, "city": "NY"}])
    assert problems == ["row 0, column age: 'abc' is not a number",
                        "row 0, column city: 'Paris' is not a known value (LA, NY, SF)",
                        "row 1, column income: has no value in this row"]
    assert modelkeep.validate(model, [{"age": 1, "income": 2}]) == ["Missing columns: city"]


def test_model_without_column_names():
    model = RandomForestClassifier(n_estimators=3, random_state=0).fit(np.arange(30).reshape(10, 3), [0, 1] * 5)
    assert [column.name for column in modelkeep.expects(model).columns] == ["x0", "x1", "x2"]
    assert modelkeep.validate(model, [[1, 2, 3]]) == []
    assert modelkeep.validate(model, [[1, 2]]) == ["The model expects 3 columns. The rows have 2."]
    assert modelkeep.validate(model, [[1, "x", 3]]) == ["row 0, column x1: 'x' is not a number"]
    assert "is missing" in modelkeep.validate(LogisticRegression().fit([[0.0], [1.0]], [0, 1]), [[float("nan")]])[0]


def test_text_model():
    texts = ["free prize now", "meeting at noon", "win money free", "lunch tomorrow"]
    model = Pipeline([("words", TfidfVectorizer()), ("model", LogisticRegression())]).fit(texts, [1, 0, 1, 0])
    assert modelkeep.expects(model).text is True
    assert modelkeep.validate(model, ["some text"]) == []
    assert modelkeep.validate(model, "one text") == ["Give a list of texts, not one text."]
    assert modelkeep.validate(model, ["ok", 3]) == ["row 1: the value is not text"]


def test_describe_an_old_file(capsys):
    description = modelkeep.describe(FIXTURES / "1.2.2" / "scaled_forest.joblib")
    assert description.saved == ["1.2.2"]
    assert description.outline[0].startswith("Pipeline")
    assert "    5 x DecisionTreeClassifier" in description.outline
    assert len(description.inputs.columns) == 4
    assert description.classes == [0, 1]
    assert main(["describe", str(FIXTURES / "1.2.2" / "scaled_forest.joblib")]) == 0
    assert "Input" in capsys.readouterr().out


def test_check_a_folder_as_json(capsys):
    import json

    assert main(["check", str(FIXTURES / "1.8.0"), "--json"]) == 1
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 35
    assert {report["status"] for report in reports} == {"current", "repairable"}
    old = modelkeep.check(FIXTURES / "1.2.2" / "tree_reg.pkl")
    assert old.old_way == "scikit-learn 1.2.2 on Python 3.8 to 3.11"
