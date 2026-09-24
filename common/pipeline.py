"""One pipeline for every dataset. All stateful steps are fitted inside the
training data they are given, never on the full dataset (reviewer R10)."""
import os
import tempfile
from pathlib import Path

from joblib import Memory
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .corrected import CorrectedClassifier

SVD_COMPONENTS = 300

# Caches the fitted preprocessing steps. The search tries 20 hyperparameter
# candidates on the same inner folds; without the cache TF-IDF + SVD would be
# refitted for every one of them. Results are identical with or without it.
# One folder per process, emptied after every outer fold (see cv.fit_evaluate).
_CACHE = Memory(Path(tempfile.gettempdir()) / f"ablation_cache_{os.getpid()}", verbose=0)


def clear_cache() -> None:
    if _CACHE is not None:
        _CACHE.clear(warn=False)


def build_pipeline(modality: str, base, sampling_ratio: float, weight_mode: str,
                   weight_power: float, seed: int) -> Pipeline:
    if modality == "text":
        steps = [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            # Dense space: SMOTE interpolation and GNB are meaningful here.
            ("svd", TruncatedSVD(n_components=SVD_COMPONENTS, random_state=seed)),
        ]
    elif modality == "tabular":
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(0.0)),
        ]
    else:
        raise ValueError(f"Unknown modality '{modality}'")

    steps += [
        ("scale", StandardScaler()),
        ("clf", CorrectedClassifier(base=base, sampling_ratio=sampling_ratio,
                                    weight_mode=weight_mode, weight_power=weight_power,
                                    random_state=seed)),
    ]
    return Pipeline(steps, memory=_CACHE)
