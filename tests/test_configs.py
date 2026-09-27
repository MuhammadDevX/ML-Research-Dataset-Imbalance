from common import configs, settings
from common.classifiers import SEARCH_SPACES
from common.cv import e2_grid


def test_full_factorial_plus_recalc_arm():
    cells = {(c["A"], c["B"], c["C"]) for k, c in configs.CONFIGS.items() if k != "c8_A_Brecalc"}
    assert cells == {(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
    assert configs.CONFIGS["c8_A_Brecalc"]["weight_mode"] == "recalc"


def test_twins_differ_only_in_threshold():
    for cid, cfg in configs.CONFIGS.items():
        if cfg["twin"]:
            twin = configs.CONFIGS[cfg["twin"]]
            assert cfg["C"] == 1 and twin["C"] == 0
            assert (cfg["A"], cfg["B"], cfg["weight_mode"]) == (twin["A"], twin["B"], twin["weight_mode"])
            assert configs.RUN_ORDER.index(cfg["twin"]) < configs.RUN_ORDER.index(cid)


def test_params_from_source_runs_first_and_uses_same_resampling():
    for cid, cfg in configs.CONFIGS.items():
        src = cfg.get("params_from")
        if src:
            assert configs.CONFIGS[src]["A"] == cfg["A"] and configs.CONFIGS[src]["C"] == cfg["C"]
            assert configs.RUN_ORDER.index(src) < configs.RUN_ORDER.index(cid)


def test_weight_mode_matches_B():
    for cfg in configs.CONFIGS.values():
        assert (cfg["weight_mode"] != "none") == bool(cfg["B"])


def test_every_classifier_has_a_search_space():
    assert set(settings.CLASSIFIERS) == set(SEARCH_SPACES)
    assert set(settings.E2_CLASSIFIERS) <= set(settings.CLASSIFIERS)


def test_e2_grid_size():
    # 8 powers x 2 C without resampling; (8 powers + recalc) x 2 C for each ratio > 0
    assert len(e2_grid()) == 8 * 2 + 2 * 9 * 2
