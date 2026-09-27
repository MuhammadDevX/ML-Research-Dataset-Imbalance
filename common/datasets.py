"""Dataset registry. To add a dataset, add one entry here (see TEAM_GUIDE.md)."""

DATASETS = {
    # ---- member 1 -------------------------------------------------------
    "oilspill": dict(
        title="Oil Spill",
        domain="remote_sensing",
        owner="member_1",
        modality="tabular",
    ),
    "smsspam": dict(
        title="SMS Spam Collection",
        domain="text_telecom",
        owner="member_1",
        modality="text",
    ),
    "cicids2017": dict(
        title="CIC-IDS2017",
        domain="network_security",
        owner="member_1",
        modality="tabular",
    ),
    # E3 pool (BENIGN vs Bot); its splits are stored per imbalance level.
    "cicids2017_e3": dict(
        title="CIC-IDS2017 IR sweep (BENIGN vs Bot)",
        domain="network_security",
        owner="member_1",
        modality="tabular",
    ),
    # ---- member 2 -------------------------------------------------------
    "aps": dict(
        title="APS Failure (Scania Trucks)",
        domain="manufacturing",
        owner="member_2",
        modality="tabular",
    ),
    "creditfraud": dict(
        title="Credit Card Fraud (ULB)",
        domain="finance",
        owner="member_2",
        modality="tabular",
    ),
}


def info(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Registered: {sorted(DATASETS)}")
    return DATASETS[dataset]
