"""Find the input that a fitted model expects, and check rows against it.

A fitted model keeps the names of its input columns. A ColumnTransformer also shows which columns are numbers,
which are categories and which values each category can have. This module reads that data from the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NUMBER, CATEGORY, TEXT, ANY = "number", "category", "text", "any"
MAX_PROBLEMS = 20
# Steps that accept only numbers. An estimator at the start of a model also accepts only numbers.
NUMBER_STEPS = {
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "PolynomialFeatures", "KBinsDiscretizer", "Binarizer", "PCA", "TruncatedSVD",
    "KNNImputer", "IterativeImputer", "VarianceThreshold", "SelectKBest", "SelectPercentile", "SplineTransformer",
}


@dataclass
class Column:
    name: str
    kind: str = NUMBER
    allowed: list | None = None     # the known values of a category
    strict: bool = False            # True: a value that is not in "allowed" stops the model
    used: bool = True               # False: the model drops this column

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "allowed": self.allowed, "strict": self.strict, "used": self.used}


@dataclass
class Contract:
    columns: list[Column] = field(default_factory=list)
    text: bool = False              # True: the input is a list of texts, not a table
    named: bool = False             # True: the model was fitted with column names
    allows_missing: bool = True

    def as_dict(self) -> dict:
        return {"text": self.text, "named": self.named, "allows_missing": self.allows_missing,
                "columns": [column.as_dict() for column in self.columns]}

    def json_schema(self) -> dict:
        """JSON Schema for one input row. Use it in an API to refuse bad requests."""
        if self.text:
            return {"type": "string"}
        properties = {}
        for column in self.columns:
            if column.kind == NUMBER:
                properties[column.name] = {"type": "number"}
            elif column.kind == CATEGORY and column.allowed is not None and column.strict:
                properties[column.name] = {"enum": column.allowed}
            elif column.kind == TEXT:
                properties[column.name] = {"type": "string"}
            else:
                properties[column.name] = {}
        return {"type": "object", "properties": properties,
                "required": [column.name for column in self.columns if column.used]}

    def __str__(self) -> str:
        if self.text:
            return "one text for each row"
        lines = []
        for column in self.columns:
            detail = column.kind
            if column.allowed is not None:
                shown = ", ".join(str(value) for value in column.allowed[:6]) + (", ..." if len(column.allowed) > 6 else "")
                detail += f": {shown}"
            if not column.used:
                detail += " (not used)"
            lines.append(f"{column.name:<24} {detail}")
        return "\n".join(lines)


def _name(obj) -> str:
    return type(obj).__name__


def _is_text_step(step) -> bool:
    return _name(step).endswith("Vectorizer") and hasattr(step, "analyzer")


def _first_step(model):
    """Go into search objects and pipelines until the first step that reads the input."""
    for _ in range(20):
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
        elif hasattr(model, "steps") and model.steps:
            model = model.steps[0][1]
        else:
            return model
    return model


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def _kind_of(step) -> tuple[str, list | None, bool]:
    """Return (kind, known values of each column, strict) for one step of a ColumnTransformer."""
    if isinstance(step, str):
        return ANY, None, False
    if hasattr(step, "steps"):
        for _, inner in step.steps:
            kind, allowed, strict = _kind_of(inner)
            if kind in (CATEGORY, TEXT):
                return kind, allowed, strict
        return _kind_of(step.steps[0][1]) if step.steps else (ANY, None, False)
    if hasattr(step, "categories_"):
        values = [[_plain(value) for value in group] for group in step.categories_]
        return CATEGORY, values, getattr(step, "handle_unknown", "error") == "error"
    if _is_text_step(step):
        return TEXT, None, False
    if _name(step) == "SimpleImputer":
        return (ANY if getattr(step, "strategy", "mean") in ("most_frequent", "constant") else NUMBER), None, False
    if _name(step) in NUMBER_STEPS or hasattr(step, "predict"):
        return NUMBER, None, False
    return ANY, None, False


def _positions(selection, names: list[str]) -> list[int]:
    """Change the column selection of a ColumnTransformer to a list of positions."""
    if isinstance(selection, str):
        return [names.index(selection)] if selection in names else []
    if isinstance(selection, slice):
        return list(range(len(names)))[selection]
    if isinstance(selection, int):
        return [selection]
    found = []
    values = selection.tolist() if hasattr(selection, "tolist") else list(selection)
    if values and all(isinstance(value, bool) for value in values):
        return [index for index, keep in enumerate(values) if keep]
    for value in values:
        if isinstance(value, str) and value in names:
            found.append(names.index(value))
        elif isinstance(value, int):
            found.append(value)
    return found


def _allows_missing(model) -> bool:
    try:
        from sklearn.utils import get_tags

        return bool(get_tags(model).input_tags.allow_nan)
    except Exception:  # noqa: BLE001 - old scikit-learn versions have no tags; do not report missing values then
        return True


def expects(model) -> Contract:
    """Return the input that the fitted model expects."""
    first = _first_step(model)
    if _is_text_step(first):
        return Contract(text=True)
    raw = getattr(model, "feature_names_in_", None)
    if raw is None:
        raw = getattr(first, "feature_names_in_", None)
    count = getattr(model, "n_features_in_", None) or getattr(first, "n_features_in_", None)
    if raw is None and count is None:
        raise ValueError(f"{_name(model)} does not say which input it expects. Is it fitted?")
    names = [str(name) for name in raw] if raw is not None else [f"x{index}" for index in range(int(count))]
    contract = Contract(named=raw is not None, allows_missing=_allows_missing(model))

    if hasattr(first, "transformers_"):
        columns = [Column(name, ANY, used=False) for name in names]
        for _, step, selection in first.transformers_:
            kind, allowed, strict = _kind_of(step)
            for order, position in enumerate(_positions(selection, names)):
                if not 0 <= position < len(columns):
                    continue
                column = columns[position]
                column.kind, column.used = kind, not (isinstance(step, str) and step == "drop")
                if allowed is not None and order < len(allowed):
                    column.allowed, column.strict = allowed[order], strict
        contract.columns = columns
        return contract

    kind, allowed, strict = _kind_of(first)
    contract.columns = [Column(name, kind) for name in names]
    if allowed is not None:
        for column, values in zip(contract.columns, allowed, strict=False):
            column.allowed, column.strict = values, strict
    return contract


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) or type(value).__module__ == "numpy" and hasattr(value, "dtype") and value.dtype.kind in "fiub"


def _rows_and_names(rows) -> tuple[list[list], list[str] | None]:
    if hasattr(rows, "columns") and hasattr(rows, "values"):
        return rows.astype(object).values.tolist(), [str(name) for name in rows.columns]
    if isinstance(rows, dict):
        rows = [rows]
    rows = rows.tolist() if hasattr(rows, "tolist") else list(rows)
    if rows and isinstance(rows[0], dict):
        names = list(dict.fromkeys(str(key) for row in rows for key in row))
        return [[{str(k): v for k, v in row.items()}.get(name, _ABSENT) for name in names] for row in rows], names
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    return [list(row) for row in rows], None


_ABSENT = object()


def validate(target, rows) -> list[str]:
    """Check rows against a model or a Contract. Return the problems as sentences. An empty list means no problem."""
    contract = target if isinstance(target, Contract) else expects(target)
    if contract.text:
        if isinstance(rows, str):
            return ["Give a list of texts, not one text."]
        bad = [index for index, item in enumerate(rows) if not isinstance(item, str)]
        return [f"row {index}: the value is not text" for index in bad[:MAX_PROBLEMS]]

    table, names = _rows_and_names(rows)
    wanted = [column.name for column in contract.columns]
    problems = []
    if names is None:
        widths = {len(row) for row in table}
        if widths != {len(wanted)}:
            return [f"The model expects {len(wanted)} columns. The rows have {', '.join(map(str, sorted(widths)))}."]
        order = list(range(len(wanted)))
    else:
        missing = [column.name for column in contract.columns if column.used and column.name not in names]
        if missing:
            problems.append("Missing columns: " + ", ".join(missing))
        extra = [name for name in names if name not in wanted]
        if extra and all(column.used for column in contract.columns):
            problems.append("Columns that the model does not know: " + ", ".join(extra))
        order = [names.index(name) if name in names else None for name in wanted]

    total = 0
    for number, row in enumerate(table):
        for column, position in zip(contract.columns, order, strict=True):
            if position is None or not column.used:
                continue
            value = row[position]
            problem = None
            if value is _ABSENT:
                problem = "has no value in this row"
            elif _is_missing(value):
                problem = None if contract.allows_missing else "is missing, and the model does not accept missing values"
            elif column.kind == NUMBER and (not _is_number(value) or isinstance(value, str)):
                problem = f"{value!r} is not a number"
            elif column.kind == TEXT and not isinstance(value, str):
                problem = f"{value!r} is not text"
            elif column.kind == CATEGORY and column.strict and column.allowed is not None and value not in column.allowed:
                shown = ", ".join(str(item) for item in column.allowed[:8])
                problem = f"{value!r} is not a known value ({shown}{', ...' if len(column.allowed) > 8 else ''})"
            if problem:
                total += 1
                if total <= MAX_PROBLEMS:
                    problems.append(f"row {number}, column {column.name}: {problem}")
    if total > MAX_PROBLEMS:
        problems.append(f"and {total - MAX_PROBLEMS} more problems")
    return problems
