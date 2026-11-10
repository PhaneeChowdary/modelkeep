"""Save small models with the scikit-learn version of this environment.

Run it one time for each old version. See tools/make_all_fixtures.sh.
Each model gets a pickle file, a joblib file and the predictions that the old version gave.
"""
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401

rng = np.random.RandomState(0)
X = rng.normal(size=(120, 4))
y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
target = X[:, 0] * 2 + X[:, 3]
X_new = rng.normal(size=(15, 4))
TEXT = ["free prize now", "meeting at noon", "win money free", "lunch tomorrow", "claim your prize", "see you at work"] * 5
TEXT_Y = [1, 0, 1, 0, 1, 0] * 5
TEXT_NEW = ["free money", "work meeting tomorrow", "prize"]

MODELS = {
    "tree_clf": (DecisionTreeClassifier(max_depth=4, random_state=0), y, "proba"),
    "tree_reg": (DecisionTreeRegressor(max_depth=4, random_state=0), target, "predict"),
    "forest_clf": (RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0), y, "proba"),
    "forest_reg": (RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0), target, "predict"),
    "extra_trees": (ExtraTreesClassifier(n_estimators=5, max_depth=4, random_state=0), y, "proba"),
    "gboost_clf": (GradientBoostingClassifier(n_estimators=8, max_depth=2, random_state=0), y, "proba"),
    "gboost_reg": (GradientBoostingRegressor(n_estimators=8, max_depth=2, random_state=0), target, "predict"),
    "hist_gboost": (HistGradientBoostingClassifier(max_iter=8, random_state=0), y, "proba"),
    "adaboost": (AdaBoostClassifier(n_estimators=5, random_state=0), y, "predict"),
    "isolation": (IsolationForest(n_estimators=5, random_state=0), None, "score"),
    "logistic": (LogisticRegression(), y, "proba"),
    "ridge": (Ridge(), target, "predict"),
    "knn": (KNeighborsClassifier(n_neighbors=3), y, "proba"),
    "svc": (SVC(probability=False), y, "decision"),
    "kmeans": (KMeans(n_clusters=3, n_init=3, random_state=0), None, "predict"),
    "scaled_forest": (Pipeline([("scale", StandardScaler()),
                                ("model", RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0))]), y, "proba"),
}
CALLS = {"proba": "predict_proba", "predict": "predict", "score": "score_samples", "decision": "decision_function"}


def main(out):
    out = Path(out) / sklearn.__version__
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, (model, labels, kind) in MODELS.items():
        model.fit(X) if labels is None else model.fit(X, labels)
        expected[name] = {"call": CALLS[kind], "result": getattr(model, CALLS[kind])(X_new).tolist()}
        (out / (name + ".pkl")).write_bytes(pickle.dumps(model, protocol=4))
        joblib.dump(model, out / (name + ".joblib"))
    text = Pipeline([("words", TfidfVectorizer()), ("model", LogisticRegression())]).fit(TEXT, TEXT_Y)
    expected["text"] = {"call": "predict_proba", "result": text.predict_proba(TEXT_NEW).tolist(), "text": True}
    (out / "text.pkl").write_bytes(pickle.dumps(text, protocol=4))
    joblib.dump(text, out / "text.joblib")
    joblib.dump(model, out / "compressed.joblib.gz", compress=3)
    expected["compressed"] = expected["scaled_forest"]
    info = {"sklearn": sklearn.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
            "X_new": X_new.tolist(), "text_new": TEXT_NEW, "models": expected}
    (out / "expected.json").write_text(json.dumps(info))
    print("saved", len(expected), "models with scikit-learn", sklearn.__version__)


if __name__ == "__main__":
    main(sys.argv[1])
