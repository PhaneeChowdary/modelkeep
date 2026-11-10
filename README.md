# modelkeep

Keep scikit-learn model files in a usable condition.

A saved model is a file that you must be able to load again next year. Four things go wrong:

| Problem | modelkeep |
| --- | --- |
| You upgrade scikit-learn and the file does not load | `load` and `fix` repair the file |
| The file loads after an upgrade, but the results are different, and nothing tells you | `save` keeps proof of the results, `verify` checks it |
| You get a model file and do not know what is in it | `check` and `describe` read it and run no code from it |
| A request sends bad input and the model stops with an unclear error | `expects` and `validate` name the bad value |

```sh
pip install modelkeep
```

It has no dependencies of its own. It uses the scikit-learn that you have installed: 1.4 or later, on Python 3.10 or later.

## Save with proof, verify after an upgrade

```python
import modelkeep

modelkeep.save(model, "model.joblib", X_test[:20])
```

This writes `model.joblib` and a small file `model.joblib.card.json`. The card holds the 20 rows, the results that the
model gave for them, and the versions of Python, scikit-learn and NumPy. Keep the two files together.

After you upgrade scikit-learn, or on another computer:

```
$ modelkeep verify model.joblib
model.joblib
  Result       The model gives the saved results.
  Saved with   scikit-learn 1.2.2, Python 3.10.13
  Now          scikit-learn 1.9.1, Python 3.13.1
  Repairs      10 repairs were necessary to load the file
    predict            same
    predict_proba      same
```

The exit code is 0 when the results are the same, so you can use `modelkeep verify models/` in a test or a deploy
step. The sample can be an array, a pandas table or a list of texts.

## Load a file that an old version saved

```
$ python -c "import pickle; pickle.load(open('old.pkl', 'rb'))"
ValueError: node array from the pickle has an incompatible dtype

$ modelkeep check old.pkl
old.pkl
  Saved with   scikit-learn 1.2.2 (pickle)
  Installed    scikit-learn 1.9.1
  Model        RandomForestClassifier > DecisionTreeClassifier
  Result       A plain load fails or gives wrong results. modelkeep can repair this file.
  Repairs
    stops a plain load  5 x DecisionTreeClassifier: tree data changed to the new layout; new field set to 0: missing_go_to_left
    wrong results       5 x DecisionTreeClassifier: class counts in the tree changed to fractions; without this, predict_proba returns counts
    default value       7 settings that the old version did not have
  Other way    install scikit-learn 1.2.2 on Python 3.8 to 3.11
  Next         modelkeep fix old.pkl

$ modelkeep fix old.pkl
```

`old.fixed.pkl` now loads with a plain `pickle.load`. The old file stays as it was. If the old file has a card, `fix`
writes the new file only when the repaired model gives the saved results.

```python
model = modelkeep.load("old.pkl")      # the repaired model, with no new file
```

`modelkeep check models/ --json` reads all model files in a folder and prints the reports as JSON.

### Some old models load and are wrong

An error is the good case, because you see it. These two changes give no error:

- A tree classifier that scikit-learn 1.3 saved loads in 1.4 and later, but `predict_proba` returns sample counts such
  as `[3.0, 0.0]`, not probabilities. This is also true for forests.
- A `TfidfVectorizer` from 1.4 or earlier loads in 1.5 and later, but its word weights are not applied.

`check` shows both as "wrong results". `load` and `fix` repair them.

### What is tested

The tests hold 315 model files that scikit-learn 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 and 1.8 saved, with the
results that each version gave. Each file must give the same results in scikit-learn 1.9.1, to 8 decimal places.

| Model | Problem in a plain load with 1.9 | modelkeep |
| --- | --- | --- |
| Decision tree, random forest, extra trees, isolation forest | error for files from 1.2 or earlier; wrong `predict_proba` for classifiers from 1.3 or earlier | repaired |
| Gradient boosting, histogram gradient boosting | error for every file from 1.8 or earlier | repaired |
| k nearest neighbors | error for files from 1.2 or earlier | repaired |
| Text pipeline with `TfidfVectorizer` | wrong results for files from 1.4 or earlier | repaired |
| Logistic regression, ridge, SVC, k-means, scalers | none | loads as it is |
| AdaBoost classifier with the SAMME.R method, the default until 1.5 | wrong results | refused |

A pipeline or a search object that holds these models is repaired too.

## Describe a file

```
$ modelkeep describe model.joblib
model.joblib
  File         joblib, 8 KB, saved with scikit-learn 1.2.2
  Proof card   no
  Parts
    Pipeline  (memory=None, verbose=False)
      StandardScaler  (with_mean=True, with_std=True, copy=True)
      RandomForestClassifier  (n_estimators=5, bootstrap=True, oob_score=False, n_jobs=None, random_state=0, +13)
        5 x DecisionTreeClassifier
  Input
    x0                       number
    x1                       number
  Classes      0, 1
```

## Check the input

A fitted model knows the names of its columns. A `ColumnTransformer` also knows which columns are numbers and which
values each category can have. modelkeep reads this from the model.

```python
print(modelkeep.expects(model))
# age                      number
# city                     category: LA, NY, SF
# income                   number

modelkeep.validate(model, [{"age": "abc", "city": "Paris", "income": 5}])
# ["row 0, column age: 'abc' is not a number",
#  "row 0, column city: 'Paris' is not a known value (LA, NY, SF)"]

modelkeep.expects(model).json_schema()     # JSON Schema for one row, for an API
```

An empty list means that the rows are correct. Rows can be a pandas table, a list of dictionaries, a list of lists or
a list of texts.

## Limits

- **AdaBoost with SAMME.R.** scikit-learn removed this method in 1.6, so the new version cannot give the old
  results. modelkeep stops with an explanation. `allow_changed=True` loads the model with the new method.
- **Models that are not in the table.** They load if scikit-learn did not change them. modelkeep fills settings that
  the old version did not have with their default values, and it lists each one. It does not know about other
  changes. Use `save` and `verify` to get proof for your own model.
- **scikit-learn 0.24.** These files load in my tests on real models, but 0.24 has no build for the computer that
  makes the test files, so no test compares their results.
- **A new file on an old scikit-learn.** modelkeep goes forward only.
- **Missing values in old trees.** Trees from before 1.3 did not accept missing values. A repaired tree sends them
  to the right branch. The old version stopped with an error.
- **Input checks.** `validate` knows names, numbers and categories. It does not know the range of a number, because
  a model does not keep it.

## Safety

Python can run any code when it loads a pickle file. `check` and `describe` read a file with stand-in classes, so no
code from the file runs. If the file names code that is not from scikit-learn, NumPy, SciPy or joblib, modelkeep
reports it and does not load the file. Use `trust=True` only for a file from a source that you trust.

This is a check of names, not a sandbox. For models from other people, use [skops](https://skops.readthedocs.io).

## How a repair works

1. **Scan.** A pickle reader gives a stand-in for each class in the file. The stand-ins keep the settings and the
   scikit-learn version of each part. Nothing is imported from scikit-learn.
2. **Read.** A second reader loads the file. It gives the new name to each class that moved. It keeps apart the
   parts that the new version does not accept, such as the node table of a tree.
3. **Repair.** Each of these parts is made again in the new format. The report lists each repair as one of four
   kinds: stops a plain load, wrong results, removed part, default value.

## Develop

```sh
uv venv && uv pip install -e . --group dev
.venv/bin/pytest                 # 352 tests
./tools/make_all_fixtures.sh     # save the test models again; uv downloads each old version
python tools/matrix.py           # one line for each old version
```

To add a scikit-learn version or a model, add it in `tools/make_all_fixtures.sh` or `tools/make_fixtures.py`. Run the
script, then the tests. A test that fails shows the next repair to write.
