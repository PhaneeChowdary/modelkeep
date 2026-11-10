"""Load a model file that an old scikit-learn version saved. Repair the parts that changed.

The loader reads the file with a pickle reader. The reader gives the new name to each class that moved.
It keeps apart the parts that the new version does not accept. After the read, the loader makes each of
these parts again in the new format. Each repair adds one Note, so the caller sees all changes.
"""
from __future__ import annotations

import importlib
import inspect
import io
import pickle
import threading
from dataclasses import dataclass

from ._scan import SAFE_MODULES, Scan, scan_bytes, unpack, version_key

BROKEN = "broken"    # a plain load stops with an error
SILENT = "silent"    # a plain load works, but the model gives different results
FILLED = "filled"    # a setting that the old version did not have gets its default value
LOST = "lost"        # the class of a part no longer exists; an empty object takes its place

OLD_LOSS_MODULES = ("sklearn.ensemble._gb_losses", "sklearn.ensemble._hist_gradient_boosting.loss")
MOVED_MODULES = {"_loss": "sklearn._loss._loss"}
OLD_LOSS_NAMES = {"deviance": "log_loss", "ls": "squared_error", "lad": "absolute_error",
                  "least_squares": "squared_error", "least_absolute_deviation": "absolute_error",
                  "binary_crossentropy": "log_loss", "categorical_crossentropy": "log_loss", "auto": "log_loss"}
_joblib_lock = threading.Lock()


class UnsafeFile(Exception):
    """The file names code outside scikit-learn, NumPy and SciPy. A load could run that code."""


class CannotRepair(Exception):
    """The new version cannot give the results of the old version for this model."""


@dataclass(frozen=True)
class Note:
    kind: str
    part: str
    text: str


class HeldTree:
    """Keeps the data of a decision tree. The loader makes the tree again in the new format."""

    def __init__(self, *args):
        self.args, self.state = args, None

    def __setstate__(self, state):
        self.state = state


class OldLoss:
    """Takes the place of a loss object. The module of that object no longer exists."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class Lost:
    """Takes the place of a part that scikit-learn removed. The model works if it does not use the part."""

    lost_name = ""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _rebuild_loss(cls, _checksum=None, state=()):
    """A compiled loss object keeps only numbers, such as a quantile. Make a new object from these numbers."""
    numbers = [value for value in (state or ()) if isinstance(value, (int, float)) and not isinstance(value, bool)]
    try:
        return cls(*numbers)
    except TypeError:
        return cls()


def _find_class(base, trust: bool):
    def find_class(self, module, name):
        if not trust and module.split(".")[0] not in SAFE_MODULES:
            raise UnsafeFile(f"The file names {module}.{name}. Load it only if you trust its source: use trust=True.")
        if (module, name) == ("sklearn.tree._tree", "Tree"):
            return HeldTree
        if module in OLD_LOSS_MODULES:
            return OldLoss
        if name.startswith("__pyx_unpickle_Cy") and module.split(".")[-1] == "_loss":
            return _rebuild_loss
        module = MOVED_MODULES.get(module, module)
        try:
            return base.find_class(self, module, name)
        except AttributeError:
            # Since 1.3, the names of distance and tree classes have 64 at the end.
            found = getattr(importlib.import_module(module), name + "64", None)
            if found is None and module.split(".")[0] == "sklearn":
                found = type(name, (Lost,), {"lost_name": f"{module}.{name}"})
            if found is None:
                raise
            return found
    return find_class


def _read(data: bytes, scan: Scan, trust: bool):
    if scan.kind == "joblib":
        import joblib
        import joblib.numpy_pickle as npk

        reader = type("Reader", (npk.NumpyUnpickler,), {"find_class": _find_class(npk.NumpyUnpickler, trust)})
        # joblib.load gets its reader class from this module name. Point the name to our reader for one call.
        with _joblib_lock:
            original = npk.NumpyUnpickler
            npk.NumpyUnpickler = reader
            try:
                return joblib.load(io.BytesIO(data))
            finally:
                npk.NumpyUnpickler = original
    reader = type("Reader", (pickle.Unpickler,), {"find_class": _find_class(pickle.Unpickler, trust)})
    return reader(io.BytesIO(data)).load()


def _objects(root) -> list:
    """Return all objects in the model that have attributes. Inner objects come first."""
    import numpy as np

    found, seen = [], set()

    def visit(obj, depth=0):
        if id(obj) in seen or depth > 300 or isinstance(obj, (str, bytes, int, float, bool, type(None))):
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children = list(obj)
        elif isinstance(obj, np.ndarray):
            children = obj.ravel().tolist() if obj.dtype == object else []
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            children = list(vars(obj).values())
        else:
            return
        for child in children:
            visit(child, depth + 1)
        if hasattr(obj, "__dict__") and not isinstance(obj, (type, np.ndarray)):
            found.append(obj)

    visit(root)
    return found


def _rebuild_tree(frozen: HeldTree, owner, saved: tuple, installed: tuple, notes: list) -> object:
    import numpy as np
    from sklearn.base import is_classifier
    from sklearn.tree._tree import Tree

    name = type(owner).__name__
    state = dict(frozen.state)
    n_features, n_classes, n_outputs = frozen.args
    blank = Tree(n_features, np.asarray(n_classes, dtype=np.intp), n_outputs)
    wanted = blank.__getstate__()["nodes"].dtype
    old = state["nodes"]
    if old.dtype != wanted:
        nodes = np.zeros(old.shape, dtype=wanted)
        for field in wanted.names:
            if field in old.dtype.names:
                nodes[field] = old[field]
        missing = [field for field in wanted.names if field not in old.dtype.names]
        state["nodes"] = nodes
        notes.append(Note(BROKEN, name, "tree data changed to the new layout; new field set to 0: " + ", ".join(missing)))
    if is_classifier(owner) and saved < (1, 4) and installed >= (1, 4):
        # Before 1.4, a tree kept the number of samples of each class. Since 1.4, it keeps the fraction.
        values = np.asarray(state["values"], dtype=np.float64)
        totals = values.sum(axis=2, keepdims=True)
        totals[totals == 0] = 1
        state["values"] = np.ascontiguousarray(values / totals)
        notes.append(Note(SILENT, name, "class counts in the tree changed to fractions; without this, predict_proba returns counts"))
    blank.__setstate__(state)
    return blank


def _fill_defaults(obj, notes: list) -> None:
    try:
        parameters = inspect.signature(type(obj).__init__).parameters
    except (TypeError, ValueError):
        return
    for name, parameter in parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name not in vars(obj) and parameter.default is not parameter.empty:
            setattr(obj, name, parameter.default)
            notes.append(Note(FILLED, type(obj).__name__, f"{name} = {parameter.default!r}"))


def _repair(obj, saved: tuple, installed: tuple, notes: list, allow_changed: bool) -> None:
    import numpy as np
    from sklearn.base import BaseEstimator

    name = type(obj).__name__
    data = vars(obj)

    if isinstance(data.get("tree_"), HeldTree):
        obj.tree_ = _rebuild_tree(obj.tree_, obj, saved, installed, notes)

    if not isinstance(obj, BaseEstimator):
        return

    # Since 1.2, the name of "base_estimator" is "estimator".
    if "base_estimator" in data:
        old = data.pop("base_estimator")
        if "estimator" not in data and old != "deprecated":
            obj.estimator = old
    if "base_estimator_" in data:
        old = data.pop("base_estimator_")
        data.setdefault("estimator_", old)

    if name == "TfidfVectorizer" and "_tfidf" in data:
        for setting in ("norm", "use_idf", "smooth_idf", "sublinear_tf"):
            if setting not in data and setting in vars(obj._tfidf):
                data[setting] = vars(obj._tfidf)[setting]
    # Since 1.5, the word weights are in idf_. An older installed version still reads _idf_diag.
    if name == "TfidfTransformer" and "_idf_diag" in data and installed >= (1, 5):
        diagonal = data.pop("_idf_diag")
        if "idf_" not in data:
            data["idf_"] = np.asarray(diagonal.diagonal(), dtype=np.float64)
            notes.append(Note(SILENT, name, "word weights moved to the new attribute idf_; without this, the weights are not applied"))

    if name == "AdaBoostClassifier" and data.get("algorithm") == "SAMME.R":
        text = "this model uses the SAMME.R method, which scikit-learn removed in 1.6; the new version gives different results"
        if not allow_changed:
            raise CannotRepair("AdaBoostClassifier: " + text + ". Train it again, or load with allow_changed=True.")
        notes.append(Note(SILENT, name, text + "; not repaired"))

    loss_name = "loss_" if "loss_" in data else "_loss"
    if isinstance(data.get(loss_name), OldLoss):
        del data[loss_name]
        if data.get("loss") in OLD_LOSS_NAMES:
            data["loss"] = OLD_LOSS_NAMES[data["loss"]]
        _fill_defaults(obj, notes)
        if "estimators_" in data and "n_trees_per_iteration_" not in data and name.startswith("GradientBoosting"):
            obj.n_trees_per_iteration_ = obj.estimators_.shape[1]
        obj._loss = obj._get_loss(sample_weight=None)
        notes.append(Note(BROKEN, name, "loss object made again with the new loss classes"))
    elif name.startswith("GradientBoosting") and "estimators_" in data and "n_trees_per_iteration_" not in data:
        obj.n_trees_per_iteration_ = obj.estimators_.shape[1]

    if name == "IsolationForest" and "estimators_" in data and "_decision_path_lengths" not in data:
        from sklearn.ensemble._iforest import _average_path_length

        pairs = [(_average_path_length(tree.tree_.n_node_samples), tree.tree_.compute_node_depths())
                 for tree in obj.estimators_]
        obj._average_path_length_per_tree, obj._decision_path_lengths = zip(*pairs, strict=True)
        notes.append(Note(BROKEN, name, "path lengths of each tree calculated; the new version saves them at fit"))

    if name.startswith("HistGradientBoosting") and "_predictors" in data and "_preprocessor" not in data:
        # Before 1.4, the model had no step that encodes categories. It reads the input with no change.
        obj._preprocessor = None
        obj._is_categorical_remapped = data.get("is_categorical_")
        notes.append(Note(BROKEN, name, "attributes that 1.4 added are set for a model with no category encoder"))
    if name == "HistGradientBoostingClassifier" and "classes_" in data and "_label_encoder" not in data:
        from sklearn.preprocessing import LabelEncoder

        obj._label_encoder = LabelEncoder().fit(obj.classes_)
        notes.append(Note(BROKEN, name, "label encoder made again from the saved classes"))

    _fill_defaults(obj, notes)


def repair_bytes(data: bytes, *, trust: bool = False, allow_changed: bool = False, scan: Scan | None = None):
    """Load and repair the model in these bytes. Return (model, notes, scan)."""
    scan = scan or scan_bytes(data)
    plain, _ = unpack(data)
    if not trust and scan.unknown_names:
        raise UnsafeFile("The file names code that is not from scikit-learn, NumPy or SciPy: "
                         + ", ".join(scan.unknown_names[:5]) + ". Load it only if you trust its source: use trust=True.")
    import sklearn

    saved = version_key(scan.versions[0]) if scan.versions else version_key(sklearn.__version__)
    if scan.versions and version_key(scan.versions[-1]) > version_key(sklearn.__version__):
        raise CannotRepair(f"The file was saved with scikit-learn {scan.versions[-1]}, which is newer than the installed "
                         f"{sklearn.__version__}. Install the newer version.")
    notes: list[Note] = []
    seen_names = {name for name in scan.names}
    model = _read(plain if scan.kind != "joblib" else data, scan, trust)
    moved = any(module in MOVED_MODULES for module, _ in seen_names)
    if moved or any(name.startswith("__pyx_unpickle_Cy") and module.split(".")[-1] == "_loss" for module, name in seen_names):
        notes.append(Note(BROKEN, "loss", "compiled loss object made again; its saved form changes between versions"))
    if any(module == "sklearn.metrics._dist_metrics" and not name.endswith(("32", "64")) and name.endswith("Distance")
           for module, name in seen_names) and saved < (1, 3):
        notes.append(Note(BROKEN, "distance", "distance class found under its new name"))
    for obj in _objects(model):
        if isinstance(obj, Lost):
            notes.append(Note(LOST, type(obj).__name__, f"{obj.lost_name} no longer exists; an empty stand-in takes its place"))
            continue
        _repair(obj, saved, version_key(sklearn.__version__), notes, allow_changed)
    if isinstance(model, HeldTree):
        raise CannotRepair("The file holds only a tree, without its estimator. modelkeep needs the estimator.")
    return model, notes, scan
