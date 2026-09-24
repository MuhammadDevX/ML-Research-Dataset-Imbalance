"""Generate every notebook from one template so all datasets and experiments
use identical cells. Never hand-edit a generated experiment notebook; change
this script (through a reviewed PR) and regenerate.

    python scripts/make_notebooks.py --dataset oilspill
    python scripts/make_notebooks.py --dataset pima          # teammate
    python scripts/make_notebooks.py --analysis

Existing preprocessing notebooks are never overwritten (they hold outputs),
unless --force-preprocessing is given.
"""
import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import configs, settings  # noqa: E402
from common.datasets import DATASETS  # noqa: E402

GITHUB = "MuhammadDevX/ML-Research-Dataset-Imbalance"
BRANCH = "main"

SETUP = f'''# ---- Setup: identical in every notebook. Do not edit. ----
import os, sys, subprocess
REPO_URL = "https://github.com/{GITHUB}.git"
BRANCH = "{BRANCH}"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_ROOT = "/content/ML-Research-Dataset-Imbalance"
    if not os.path.isdir(REPO_ROOT):
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_ROOT], check=True)
    else:
        subprocess.run(["git", "-C", REPO_ROOT, "pull", "--ff-only"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    f"{{REPO_ROOT}}/requirements.txt"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
    # Results go to Google Drive so they survive a disconnect.
    os.environ.setdefault("RESULTS_ROOT", "/content/drive/MyDrive/ML-Research-Dataset-Imbalance")
else:
    REPO_ROOT = os.path.abspath(os.getcwd())
    while not os.path.isdir(os.path.join(REPO_ROOT, "common")):
        parent = os.path.dirname(REPO_ROOT)
        if parent == REPO_ROOT:
            raise RuntimeError("Open this notebook from inside the repository.")
        REPO_ROOT = parent

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
from common.env import check_environment
check_environment()
'''

PREPROCESS = {
    "oilspill": [
        ("md", """## Load
`oil_spill.csv`: 937 ocean-surface patches from satellite radar images, 49 numeric features.
The label column `class` holds `'1'` (oil slick) and `'-1'` (look-alike)."""),
        ("code", """import numpy as np
import pandas as pd
from common import data, settings
from common.paths import RAW_DIR

raw = pd.read_csv(RAW_DIR / "oil_spill.csv")
print(raw.shape)
raw["class"].value_counts()"""),
        ("md", """## Clean (stateless steps only)
* Label: strip quotes, `1` -> 1 (minority, oil slick), `-1` -> 0.
* Drop `attr1`: it is the patch number (1-352), an identifier rather than a measurement.
* Drop `attr23`: constant in every row.
* Remove duplicate rows (there are none, checked below).

Imputation and scaling are **not** done here; they are fitted inside each training fold."""),
        ("code", """df = raw.copy()
df["y"] = df.pop("class").astype(str).str.strip("'\\" ").map({"1": 1, "-1": 0}).astype(int)
constant = [c for c in df.columns if c != "y" and df[c].nunique() == 1]
print("constant columns:", constant)
df = df.drop(columns=["attr1"] + constant)
print("duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)
df["y"].value_counts()"""),
        ("md", "## Save processed data and the 5x2 outer folds"),
        ("code", """data.save_processed("oilspill", df)
data.make_outer_splits(df["y"], "oilspill")
data.summary("oilspill")"""),
    ],
    "smsspam": [
        ("md", """## Load
UCI SMS Spam Collection: 5,574 English SMS messages labelled ham or spam (tab-separated)."""),
        ("code", """import zipfile
import pandas as pd
from common import data, settings
from common.paths import RAW_DIR

with zipfile.ZipFile(RAW_DIR / "sms+spam+collection.zip") as z:
    lines = z.read("SMSSpamCollection").decode("utf-8", errors="replace").splitlines()
raw = pd.DataFrame([l.split("\\t", 1) for l in lines if l.strip()], columns=["label", "text"])
print(raw.shape)
raw["label"].value_counts()"""),
        ("md", """## Clean (stateless steps only)
* Label: spam -> 1 (minority), ham -> 0.
* Remove exact duplicate messages **before** splitting. Otherwise the same message
  can sit in both a training and a test fold, which is leakage (reviewer R10).
* Remove messages that appear with both labels (none expected).

TF-IDF and SVD are **not** fitted here; they are fitted inside each training fold."""),
        ("code", """df = raw.assign(y=(raw["label"] == "spam").astype(int))[["text", "y"]]
print("duplicate messages:", df.duplicated().sum())
df = df.drop_duplicates()
conflict = df.duplicated("text", keep=False)
print("messages with both labels:", conflict.sum())
df = df[~conflict].reset_index(drop=True)
print(df.shape)
df["y"].value_counts()"""),
        ("md", "## Save processed data and the 5x2 outer folds"),
        ("code", """data.save_processed("smsspam", df)
data.make_outer_splits(df["y"], "smsspam")
data.summary("smsspam")"""),
    ],
    "cicids2017": [
        ("md", """## Pass 1: index every flow (stateless steps only)
The 8 `MachineLearningCVE` CSVs of CIC-IDS2017 (about 2.83M flows, 78 features).
Holding all of them in memory needs more than 4 GB, so this runs in two passes:
pass 1 keeps only a 64-bit hash of each flow's features plus its label, which is
enough to find duplicates and draw the samples; pass 2 re-reads the files and keeps
only the selected flows. Runs locally in about 2 minutes (raw data is not on Colab).

* Strip the leading spaces in column names.
* Drop `Fwd Header Length.1`, an exact copy of `Fwd Header Length`.
* `inf` -> NaN (imputed later inside each training fold).
* Remove duplicate flows (same features and label, across all files), then flows
  whose features appear with both labels.
* Target: BENIGN -> 0, any attack -> 1. The attack name is kept as `meta_label`
  for description only; it is never a feature."""),
        ("code", """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import data, settings
from common.paths import RAW_DIR

files = sorted((RAW_DIR / "CIC-IDS2017").glob("*.csv"))
header = pd.read_csv(files[0], nrows=0, encoding="latin-1").columns
label_raw = [c for c in header if c.strip() == "Label"][0]
dtypes = {c: "float64" for c in header if c != label_raw}


def read_clean(f):
    d = pd.read_csv(f, encoding="latin-1", dtype=dtypes)
    d.columns = d.columns.str.strip()
    d = d.drop(columns=["Fwd Header Length.1"]).rename(columns={"Label": "meta_label"})
    # The raw files spell "Web Attack – Brute Force" with a broken dash character.
    d["meta_label"] = (d["meta_label"].astype(str).str.replace("\\ufffd", "-", regex=False)
                       .str.replace("ï¿½", "-", regex=False).str.strip())
    feats = d.columns.drop("meta_label")
    n_inf = int(np.isinf(d[feats].to_numpy()).sum())
    d[feats] = d[feats].mask(np.isinf(d[feats]))
    return d, list(feats), n_inf


index, n_inf = [], 0
for i, f in enumerate(files):
    d, feat_cols, k = read_clean(f)
    n_inf += k
    index.append(pd.DataFrame({
        "file": i, "row": np.arange(len(d)),
        "fhash": pd.util.hash_pandas_object(d[feat_cols], index=False).to_numpy(),
        "label": d["meta_label"].to_numpy()}))
    print(f"{f.name}: {len(d):,} flows")
    del d
index = pd.concat(index, ignore_index=True)
print(f"raw flows: {len(index):,}, inf values: {n_inf:,}, features: {len(feat_cols)}")"""),
        ("code", """n_raw = len(index)
index = index[~index.duplicated(subset=["fhash", "label"])]
print(f"duplicate flows removed: {n_raw - len(index):,}")
conflict = index["fhash"].duplicated(keep=False)
print(f"flows with conflicting labels removed: {int(conflict.sum()):,}")
index = index[~conflict].reset_index(drop=True)
index["y"] = (index["label"] != "BENIGN").astype(int)
print(f"after cleaning: {len(index):,} flows, attack rate {index['y'].mean():.2%}")
index["label"].value_counts()"""),
        ("md", f"""## Select the rows
* **Main dataset:** stratified 50,000-flow sample; keeps the natural attack rate.
* **E3 pool (BENIGN vs Bot):** all Bot flows plus {settings.E3_N:,} random BENIGN flows.

Both samples are fixed by `MASTER_SEED`."""),
        ("code", """main_idx, _ = train_test_split(np.arange(len(index)), train_size=50_000,
                               stratify=index["y"], random_state=settings.MASTER_SEED)
main_sel = index.iloc[np.sort(main_idx)]
bot_sel = index[index["label"] == "Bot"]
benign_sel = index[index["label"] == "BENIGN"].sample(n=settings.E3_N, random_state=settings.MASTER_SEED)
e3_sel = pd.concat([benign_sel, bot_sel])
print(f"main: {len(main_sel):,} flows; E3 pool: {len(benign_sel):,} BENIGN + {len(bot_sel):,} Bot")"""),
        ("md", "## Pass 2: read the selected flows and save"),
        ("code", """def collect(sel):
    \"\"\"Read the selected (file, row) flows, returned in the order of `sel`.\"\"\"
    parts = []
    for i, f in enumerate(files):
        rows = sel.loc[sel["file"] == i, "row"].to_numpy()
        if len(rows):
            d, _, _ = read_clean(f)
            part = d.iloc[rows]
            part.index = pd.MultiIndex.from_arrays([np.full(len(rows), i), rows])
            parts.append(part)
            del d
    order = pd.MultiIndex.from_arrays([sel["file"].to_numpy(), sel["row"].to_numpy()])
    out = pd.concat(parts).loc[order].reset_index(drop=True)
    out["y"] = (out["meta_label"] != "BENIGN").astype(int)
    return out[feat_cols + ["meta_label", "y"]]


main = collect(main_sel)
data.save_processed("cicids2017", main)
data.make_outer_splits(main["y"], "cicids2017")
print(main["meta_label"].value_counts())
data.summary("cicids2017")"""),
        ("md", f"""## E3 pool: BENIGN vs Bot at controlled imbalance
Each level uses N = {settings.E3_N:,} rows with minority share {settings.E3_LEVELS}.
Levels are nested (the rows of a smaller minority share are a subset of the larger
one), so only the imbalance changes between levels."""),
        ("code", """rng = np.random.default_rng(settings.MASTER_SEED)
pool = collect(e3_sel)
data.save_processed("cicids2017_e3", pool)

pos = rng.permutation(np.flatnonzero(pool["y"] == 1))
neg = rng.permutation(np.flatnonzero(pool["y"] == 0))
for level in settings.E3_LEVELS:
    n_min = round(level * settings.E3_N)
    assert n_min <= len(pos), f"not enough Bot flows for level {level}"
    subset = np.sort(np.concatenate([pos[:n_min], neg[:settings.E3_N - n_min]]))
    data.make_outer_splits(pool["y"].to_numpy()[subset], "cicids2017_e3", level=level, subset=subset)
    print(f"level {level:.0%}: {n_min} Bot + {settings.E3_N - n_min} BENIGN")"""),
    ],
}

TEMPLATE_PREPROCESS = [
    ("md", """## Load
TODO: describe the source file, rows, features and label."""),
    ("code", """import numpy as np
import pandas as pd
from common import data, settings
from common.paths import RAW_DIR

raw = pd.read_csv(RAW_DIR / "TODO.csv")
print(raw.shape)"""),
    ("md", """## Clean (stateless steps only)
Rules (see TEAM_GUIDE.md): label 1 = minority, 0 = majority; remove duplicates
before splitting; no imputation, scaling or anything else that is fitted."""),
    ("code", """df = raw.copy()
# TODO: create df["y"] (0/1), drop identifiers, turn invalid values into NaN
df = df.drop_duplicates().reset_index(drop=True)
df["y"].value_counts()"""),
    ("md", "## Save processed data and the 5x2 outer folds"),
    ("code", """DATASET = "TODO"
data.save_processed(DATASET, df)
data.make_outer_splits(df["y"], DATASET)
data.summary(DATASET)"""),
]


def colab_badge(rel_path: str) -> str:
    url = f"https://colab.research.google.com/github/{GITHUB}/blob/{BRANCH}/{rel_path}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def notebook(cells, rel_path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    out = []
    for kind, src in cells:
        src = src.replace("{BADGE}", colab_badge(rel_path))
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = out
    return nb


def write(rel_path, cells, overwrite=True):
    path = ROOT / rel_path
    if path.exists() and not overwrite:
        print(f"kept      {rel_path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook(cells, rel_path), path)
    print(f"written   {rel_path}")


SMOKE_CELL = '''# Quick test run (2 folds, fewer classifiers and candidates): change the next line
# to  SMOKE = True . Its results go to a separate *__SMOKE.csv file and must never
# be used in the paper. Leave it as it is for the real run.
SMOKE = os.environ.get("SMOKE", "0") == "1"'''

RUN_NOTES = """**How to run on Colab**
1. Click the badge, then *Runtime > Run all*. Allow Google Drive access when asked.
2. Results are appended to Google Drive after every classifier/fold:
   `MyDrive/ML-Research-Dataset-Imbalance/{path}`.
3. If the runtime disconnects, just *Run all* again: finished folds are skipped.
4. When it is complete, copy the CSV into the same path in the repository and commit it."""


def experiment_cells(ds, cid):
    cfg = configs.CONFIGS[cid]
    title = DATASETS[ds]["title"]
    twin = (f"\n\nThis is the C = 1 twin of `{cfg['twin']}`. If that notebook has already run, its tuned "
            "hyperparameters are reused, so the threshold is the only difference. Otherwise the same "
            "seeded search runs here and gives the same hyperparameters.") if cfg["twin"] else ""
    res = f"{ds}/{cid}/results/{ds}__{cid}.csv"
    return [
        ("md", f"""# {title} - `{cid}`
{{BADGE}}

{configs.DESCRIPTIONS[cid]}

| A: SMOTETomek | B: class weights | C: threshold tuning | weight mode |
|---|---|---|---|
| {"yes" if cfg["A"] else "no"} | {"yes" if cfg["B"] else "no"} | {"yes" if cfg["C"] else "no"} | `{cfg["weight_mode"]}` |

Protocol (identical for every dataset and config, defined in `common/`): 5x2 repeated stratified
outer CV; inside each outer training fold a {settings.N_ITER}-candidate random search with
{settings.INNER_FOLDS}-fold inner CV, optimising macro-F1; {len(settings.CLASSIFIERS)} classifiers
({", ".join(settings.CLASSIFIERS)}). Imputation, scaling, resampling and weighting are all fitted
inside the training data only.{twin}

{RUN_NOTES.format(path=res)}"""),
        ("code", SETUP),
        ("code", f'DATASET = "{ds}"\nCONFIG_ID = "{cid}"\n' + SMOKE_CELL),
        ("code", """from common.cv import run_config
results = run_config(DATASET, CONFIG_ID, smoke=SMOKE)"""),
        ("md", "## Summary (mean ± SD over the 10 outer folds)"),
        ("code", """from common.report import summarize
summarize(results)"""),
    ]


def e2_cells(ds):
    title = DATASETS[ds]["title"]
    res = f"{ds}/e2_weight_sensitivity/results/{ds}__e2_weight_sensitivity.csv"
    return [
        ("md", f"""# {title} - E2 weight sensitivity
{{BADGE}}

Answers reviewer R3 (one weighting configuration only) and tests the over-correction mechanism.

* Classifiers: {", ".join(settings.E2_CLASSIFIERS)}.
* Resampling ratio: {settings.E2_SAMPLING_RATIOS} (0 = none; otherwise SMOTETomek to that minority:majority ratio).
* Positive-class weight = IR ** power, power in {settings.E2_WEIGHT_POWERS}
  (0 = none, 0.5 = sqrt-inverse, 1 = inverse frequency), plus weights recomputed after resampling.
* Each setting with and without threshold tuning (C).
* Hyperparameters are reused from `c0_none` (no resampling) and `c1_A_smotetomek` (with resampling).
  **Run those two notebooks first**; missing folds are tuned here, which is slower.
* Each row records `rho_eff`, the positive:negative loss mass the classifier sees.

{RUN_NOTES.format(path=res)}"""),
        ("code", SETUP),
        ("code", f'DATASET = "{ds}"\n' + SMOKE_CELL),
        ("code", """from common.cv import run_e2
results = run_e2(DATASET, smoke=SMOKE)"""),
        ("md", "## Macro-F1 by resampling ratio, weight setting and threshold tuning"),
        ("code", """results.pivot_table(index=["sampling_ratio", "weight_mode", "weight_power"],
                    columns=["classifier", "C"], values="macro_f1").round(4)"""),
        ("code", """results.groupby(["sampling_ratio", "weight_mode", "weight_power"])[["rho_eff", "precision_1", "recall_1", "ppr_ratio"]].mean().round(3)"""),
    ]


def e3_cells():
    res = "cicids2017/e3_ir_sweep/results/cicids2017__e3_ir_sweep.csv"
    return [
        ("md", f"""# CIC-IDS2017 - E3 controlled imbalance sweep (BENIGN vs Bot)
{{BADGE}}

Only the imbalance ratio changes: N = {settings.E3_N:,} flows at minority share {settings.E3_LEVELS}.
All 9 configs are run with {", ".join(settings.E3_CLASSIFIERS)} and {settings.N_ITER_E3} search candidates
per fold (reduced budget because this is 5 levels x 9 configs).

To split the work across Colab sessions, set `LEVELS` to a subset, e.g. `[0.15, 0.10]` in one
session and `[0.05, 0.02, 0.01]` in another. Both append to the same file.

{RUN_NOTES.format(path=res)}"""),
        ("code", SETUP),
        ("code", "LEVELS = None   # None = all levels\n" + SMOKE_CELL),
        ("code", """from common.cv import run_e3
results = run_e3(LEVELS, smoke=SMOKE)"""),
        ("md", "## Macro-F1 by imbalance level and config (mean over classifiers and folds)"),
        ("code", """results.pivot_table(index="config_id", columns="ir_level", values="macro_f1").round(4)"""),
    ]


def meta_cells(ds):
    title = DATASETS[ds]["title"]
    return [
        ("md", f"""# {title} - dataset meta-features
{{BADGE}}

Size, imbalance ratio and class-overlap measures (F1 Fisher ratio, N1, N3) for Table 1 and the
exploratory analysis of when each correction helps (reviewers R1, R4). Descriptive only; never
used for model selection. Fast: runs in about a minute."""),
        ("code", SETUP),
        ("code", f'DATASET = "{ds}"'),
        ("code", """import pandas as pd
from common import data
from common.complexity import meta_features
from common.datasets import info
from common.paths import REPO_ROOT

X, y = data.load_processed(DATASET)
mf = meta_features(X, y, info(DATASET)["modality"], DATASET)
out = REPO_ROOT / DATASET / "meta_features" / f"{DATASET}__meta_features.csv"
mf.to_csv(out, index=False)
print("saved", out.relative_to(REPO_ROOT))
mf.T"""),
    ]


def preprocessing_cells(ds):
    title = DATASETS[ds]["title"]
    body = PREPROCESS.get(ds, TEMPLATE_PREPROCESS)
    return [("md", f"""# {title} - preprocessing
{{BADGE}}

Stateless steps only: label mapping, removing identifiers and duplicates, marking invalid values
as missing, and (for large data) a fixed stratified sample. Everything that is *fitted*
(imputation, scaling, TF-IDF, resampling) happens later inside each training fold.

Writes `data/processed/{ds}.parquet` and the fixed 5x2 outer folds `data/splits/{ds}_outer.npz`.
Both are committed, so every experiment notebook (and your teammate) uses identical data and folds.
Raw data is expected in `data/raw/` (not committed; see `data/README.md`).""")] + [
        ("code", SETUP)] + body


def analysis_cells():
    return [
        ("md", """# Merge all results
{BADGE}

Collects every `*/*/results/*.csv` (smoke files excluded), checks the schema, and reports how
complete each experiment is. Run it before every pull request."""),
        ("code", SETUP),
        ("code", """from common.report import merge_results, completeness
from common.paths import results_root, REPO_ROOT

df = merge_results(results_root())
print(f"{len(df):,} rows from", results_root())
report = completeness(df)
report"""),
        ("code", """incomplete = report[~report["complete"]]
print("incomplete experiments:", len(incomplete))
print("git commits used:", sorted(set(df["git_commit"].astype(str))))
out = REPO_ROOT / "analysis" / "tables" / "all_results.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("saved", out.relative_to(REPO_ROOT))"""),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--force-preprocessing", action="store_true")
    a = ap.parse_args()

    for ds in a.dataset:
        if ds not in DATASETS or ds == "cicids2017_e3":
            raise SystemExit(f"Register '{ds}' in common/datasets.py first.")
        write(f"{ds}/00_preprocessing/{ds}_preprocessing.ipynb", preprocessing_cells(ds),
              overwrite=a.force_preprocessing)
        write(f"{ds}/meta_features/{ds}_meta_features.ipynb", meta_cells(ds))
        for cid in configs.CONFIGS:
            write(f"{ds}/{cid}/{ds}_{cid}.ipynb", experiment_cells(ds, cid))
        write(f"{ds}/e2_weight_sensitivity/{ds}_e2_weight_sensitivity.ipynb", e2_cells(ds))
        if ds == "cicids2017":
            write("cicids2017/e3_ir_sweep/cicids2017_e3_ir_sweep.ipynb", e3_cells())
    if a.analysis:
        write("analysis/01_merge_results.ipynb", analysis_cells())


if __name__ == "__main__":
    main()
