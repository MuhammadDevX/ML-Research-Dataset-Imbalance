"""The 2^3 factorial (A, B, C) plus the recomputed-weights arm.

A = SMOTETomek resampling, B = cost-sensitive weighting, C = decision-threshold
tuning. The classifier is a separate axis and is never a "strategy".

`twin` is the C = 0 config that has exactly the same pipeline. A C = 1 config
reuses its twin's tuned hyperparameters when the twin's results exist, so the
only difference between the two is the threshold.
"""

CONFIGS = {
    "c0_none":                    dict(A=0, B=0, C=0, weight_mode="none",   twin=None),
    "c1_A_smotetomek":            dict(A=1, B=0, C=0, weight_mode="none",   twin=None),
    "c2_B_weights":               dict(A=0, B=1, C=0, weight_mode="orig",   twin=None),
    "c3_C_threshold":             dict(A=0, B=0, C=1, weight_mode="none",   twin="c0_none"),
    "c4_AB_smotetomek_weights":   dict(A=1, B=1, C=0, weight_mode="orig",   twin=None),
    "c5_AC_smotetomek_threshold": dict(A=1, B=0, C=1, weight_mode="none",   twin="c1_A_smotetomek"),
    "c6_BC_weights_threshold":    dict(A=0, B=1, C=1, weight_mode="orig",   twin="c2_B_weights"),
    "c7_ABC_full":                dict(A=1, B=1, C=1, weight_mode="orig",   twin="c4_AB_smotetomek_weights"),
    "c8_A_Brecalc":               dict(A=1, B=1, C=0, weight_mode="recalc", twin=None),
}

# Twins first, so the C = 1 configs can reuse their hyperparameters.
RUN_ORDER = [
    "c0_none", "c1_A_smotetomek", "c2_B_weights", "c4_AB_smotetomek_weights",
    "c8_A_Brecalc", "c3_C_threshold", "c5_AC_smotetomek_threshold",
    "c6_BC_weights_threshold", "c7_ABC_full",
]

DESCRIPTIONS = {
    "c0_none": "No correction (baseline).",
    "c1_A_smotetomek": "Resampling only: SMOTETomek to 1:1 on the training fold.",
    "c2_B_weights": "Weights only: positive-class weight = IR of the training fold.",
    "c3_C_threshold": "Threshold tuning only: decision threshold tuned for macro-F1 on inner folds.",
    "c4_AB_smotetomek_weights": "Resampling + weights computed from the ORIGINAL ratio (v1 'Test 1').",
    "c5_AC_smotetomek_threshold": "Resampling + threshold tuning.",
    "c6_BC_weights_threshold": "Weights + threshold tuning.",
    "c7_ABC_full": "Resampling + original-ratio weights + threshold tuning (full hybrid).",
    "c8_A_Brecalc": "Resampling + weights RECOMPUTED on the resampled data (reviewer R7).",
}


def get(config_id: str) -> dict:
    if config_id not in CONFIGS:
        raise KeyError(f"Unknown config '{config_id}'. Valid: {list(CONFIGS)}")
    return CONFIGS[config_id]


def sampling_ratio(cfg: dict) -> float:
    """Minority/majority ratio SMOTETomek targets; 0.0 means no resampling."""
    return 1.0 if cfg["A"] else 0.0


def weight_power(cfg: dict) -> float:
    return 1.0 if cfg["B"] else 0.0
