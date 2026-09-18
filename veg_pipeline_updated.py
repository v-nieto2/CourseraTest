"""
veg_pipeline_functions.py
=========================

Function library for the ComEd vegetation-manhours unitization pipeline
(reproduces IC_training.ipynb as importable, globals-free functions).

Leakage rules encoded here
--------------------------
* Everything that is *fit* (cluster objects, category vocabularies, imputation medians,
  feature selection, hyperparameters, tier boundaries) is fit on Train or on
  feeder-grouped CV over Train+Validation only.
* Early stopping inside CV uses a feeder slice of the TRAINING fold, never the scored fold.
* Resampling touches only the fitting rows of a fold.
* Test is confirmation-only. Holdout is scored once, behind a flag in the driver script.

Model structure (v3)
--------------------
* Model UNITS: Accessible (all areas) + one inaccessible unit per area class, each with its own
  permitted loss families (see ``default_model_units``).
* Tier SCHEMES are interchangeable: 4 hour tiers (cuts on hours, per accessibility regime) or
  4 volume tiers (Train quartiles of ``bay_total_intrusion_volume_(cubic_ft)``).
* Under-performing tiers are handled with SAMPLE WEIGHTS (strength searched from 0, so CV can
  reject weighting - it hurt in an earlier iteration) instead of over/under-sampling.
* Final reporting: 24 bid-time classes (2 accessibility x 3 area x 4 tiers) with a risk score.

Drivers: ``run_veg_pipeline.ipynb`` (CatBoost) and ``nn_veg_workflow.ipynb`` (PyTorch).
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from scipy.stats import bootstrap
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # the NN workflow only needs the metric/IO helpers
    CatBoostRegressor = Pool = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONSTANTS
# =============================================================================
RANDOM_STATE = 42
TARGET = "manhours"
SEGMENTS: Dict[str, int] = {"Accessible": 0, "Inaccessible": 1}
TIER_LABELS = ["Light", "Light-Medium", "Medium-Heavy", "Heavy"]
AREA_CLASSES = ["Urban", "Suburban", "Rural"]
SPLIT_NAMES = ["train", "test", "valid", "holdout"]

# Removed as features in v3; dropped on load if an older split file still carries them.
LEGACY_TIER_FEATURE_COLS = ["vegetation_tier_kmeans", "veg_tier_hierarchical"]
VOLUME_COL = "bay_total_intrusion_volume_(cubic_ft)"
TIER_SCHEMES = ["hours", "volume"]
AREA_CANDIDATE_COLS = ["dominant_land_cover_general", "area_lc", "area_lcxfm", "area_xfmr"]
LAND_COVER_NUMERIC_FEATURES = [
    "percent_forest_coverage", "percent_developed_coverage", "percent_wetland_coverage",
    "land_cover_diversity", "land_cover_pixel_count",
]
LAND_COVER_CATEGORICAL_FEATURES = ["dominant_land_cover_type"]

# Provisional cutoffs from the notebook's Section 3 (derived from full data -> replaced by Section 12).
PROVISIONAL_TIER_CUTS = {"Accessible": [2.0, 3.0, 4.0], "Inaccessible": [2.5, 4.5, 7.0]}

# TODO(veg-team/GIS): confirm. Values containing urban/suburban/rural map automatically.
AREA_VALUE_MAP: Dict[str, Dict[str, str]] = {
    "dominant_land_cover_general": {},
    "area_lc": {},
    "area_lcxfm": {},
    "area_xfmr": {},
}

REQUESTED_NUMERIC_ROAD_FEATURES = [
    "multiple_nearest_roads_tied", "nearest_road_distance_ft", "nearest_road_aadt",
    "nearest_road_aadt_year", "nearest_road_speed_limit", "nearest_road_urban",
    "nearest_road_access_control", "nearest_road_lane_count", "nearest_road_lane_width",
    "nearest_road_surface_width", "nearest_road_distance_log1p", "road_within_25ft",
    "road_within_50ft", "road_within_100ft", "road_within_150ft", "road_within_250ft",
    "road_within_500ft", "road_within_1000ft", "road_span_bearing_difference_deg",
    "road_parallel_10deg", "road_parallel_15deg", "road_parallel_20deg", "road_parallel_30deg",
    "span_length_within_road_25ft", "pct_span_within_road_25ft", "span_length_within_road_50ft",
    "pct_span_within_road_50ft", "span_length_within_road_100ft", "pct_span_within_road_100ft",
    "offroad_span_length_50ft", "road_follows_span_50pct", "road_follows_span_75pct",
    "road_follows_span_80pct", "road_follows_span_90pct", "road_follows_span",
    "road_adjacent_majority", "nearest_road_aadt_log1p", "nearest_road_is_urban",
    "nearest_road_has_access_control", "nearest_road_is_local", "nearest_road_is_collector",
    "nearest_road_is_arterial", "nearest_road_is_interstate", "nearest_road_is_high_traffic",
    "nearest_road_missing", "nearest_road_attribute_missing", "nearest_road_aadt_missing",
    "nearest_road_distance_zero", "span_zero_length", "road_zero_length",
    "roadside_workability_score", "road_access_score", "offroad_span_share_50ft",
    "nearest_road_jurisdiction", "nearest_road_func_class",
    # alley features (recently added)
    "nearest_alley_ft", "multiple_nearest_alleys_tied", "alley_span_bearing_difference_deg",
    "parallel_to_alley", "alley_within_25ft", "alley_within_50ft",
] + LAND_COVER_NUMERIC_FEATURES

REQUESTED_CATEGORICAL_ROAD_FEATURES = [
    "road_distance_tier", "road_access_category", "nearest_road_fc_name",
] + LAND_COVER_CATEGORICAL_FEATURES + AREA_CANDIDATE_COLS

# Post-work / target-derived name fragments that must never be model inputs
# ("veg_tier" also catches the removed cluster-tier columns).
LEAKY_NAME_PATTERNS = ["manhour", "crew_hrs", "total_trims", "total_removals", "brush_",
                       "total_veg_work", "complete_date", "veg_tier"]

CB_FIXED = {"random_seed": RANDOM_STATE, "allow_writing_files": False, "verbose": False, "thread_count": -1}


def default_dev_cfg(run_mode: str = "full") -> dict:
    fast = run_mode == "fast"
    return {
        "n_folds": 3 if fast else 5,
        "es_feeder_share": 0.15,
        "early_stopping_rounds": 50 if fast else 100,
        "max_iterations": 300 if fast else 3000,
        "n_trials_road": 2 if fast else 35,
        "n_trials_reg": 4 if fast else 40,
        "n_trials_weighted": 4 if fast else 25,
        "min_unit_spans": 150 if fast else 1500,   # below this a unit trains on its whole accessibility regime
        "underpred_ratio_tol": 0.03,               # sum(pred)/sum(actual) < 0.97 -> "continued underprediction"
        "underpred_rate_tol": 0.55,                # or > 55% of spans under-predicted
        "multi_quantile_alphas": [0.10, 0.50, 0.90],
        "report_test_for_experiments": True,
        "w_span_mae": 0.5,
        "w_feeder_wape": 0.5,
        "show_progress": not fast,
    }


# =============================================================================
# 1. OUTPUT HELPERS (script-friendly display / figures / json)
# =============================================================================
def show(obj, title: Optional[str] = None, digits: int = 4, max_rows: int = 60):
    """display() in notebooks, print() in scripts."""
    if title:
        print(f"\n{title}")
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        try:
            obj = obj.round(digits)
        except TypeError:          # mixed object columns (lists, strings) - show as is
            pass
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            display(obj)
            return
    except ImportError:
        pass
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        with pd.option_context("display.max_rows", max_rows, "display.width", 250,
                               "display.max_columns", 40):
            print(obj)
    else:
        print(obj)


def finish_figure(fig, save_path: Optional[str] = None, show_plot: bool = False):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# =============================================================================
# 2. LOAD & CLEAN
# =============================================================================
def load_span_data(input_dir: str) -> pd.DataFrame:
    """Read bay_clean_data.csv, make a unique bay key, recode work type."""
    df = pd.read_csv(os.path.join(input_dir, "bay_clean_data.csv"))
    nonunique = set(df["bay"].value_counts()[lambda s: s > 1].index)
    print(f"Duplicate bay records found: {len(nonunique)}")

    idx_in_group = df.groupby("bay").cumcount() + 1
    is_dup = df["bay"].isin(nonunique)
    df["bay_unique"] = np.where(is_dup, df["bay"] * 100 + idx_in_group, df["bay"])
    df["duplicate"] = is_dup.astype(int)

    # Lift-related = 1, Unavailable = 0, No Lift Access = 2
    # NOTE: confirm bay_work_type is known *before* work starts; otherwise this is post-work information.
    work_type = df["bay_work_type"].astype(str)
    df["bay_worktype_n"] = np.select(
        [work_type.str.contains("Unavailable"), work_type.str.contains("No Lift Access")], [0, 2], default=1
    )

    df["complete_date_raw"] = df["complete_date"].copy()
    df["complete_date"] = pd.to_datetime(df["complete_date_raw"], format="mixed", errors="coerce")
    df["date_only"] = df["complete_date"].dt.date
    df[TARGET] = df["total_crew_hrs"].round(2)
    print(f"Spans: {len(df):,} | unique completion days: {df['date_only'].dropna().nunique()}")
    return df


def join_span_aggregates(df: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    agg = pd.read_csv(os.path.join(input_dir, "bay_span_agg_data.csv"))
    cols = ["truck_access", "num_circuits", "num_conductors", "num_phases_cegis", "aerial_cable", "bay"]
    df = df.drop(columns=[c for c in cols if c in df.columns and c != "bay"])
    df = df.merge(agg[cols].drop_duplicates("bay"), on="bay", how="left")

    truck_map = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    df["truck_access"] = (df["truck_access"].astype(str).str.strip().str.lower()
                          .map(truck_map).fillna(-1).astype(int))
    for col in ["num_circuits", "num_conductors", "num_phases_cegis", "aerial_cable"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1)
    return df


def merge_accessibility(df: pd.DataFrame, input_dir: str,
                        filename: str = "Matteo_model_all_images_predictions_with_railroads.csv",
                        unknown_truck_as_accessible: bool = True) -> pd.DataFrame:
    """Pole-level accessibility classifier -> span flags (either pole flagged => flagged)."""
    access = pd.read_csv(os.path.join(input_dir, filename))
    missing = {"pole_number", "output", "has_railroad"} - set(access.columns)
    if missing:
        raise ValueError(f"Missing required accessibility columns: {sorted(missing)}")

    access["client_pole_id"] = access["pole_number"].astype(str).str.strip()
    access["inaccessible_flag"] = access["output"].astype(str).str.strip().eq("3")
    access["railroad_flag"] = access["has_railroad"].fillna(0).astype(int).eq(1)
    lookup = (access[["client_pole_id", "inaccessible_flag", "railroad_flag"]]
              .drop_duplicates("client_pole_id", keep="last"))

    p1, p2 = "bay_client_pole1_id", "bay_client_pole2_id"
    for col in (p1, p2):
        if col not in df.columns:
            raise ValueError(f"Expected column not found: {col}")
        df[col] = df[col].astype(str).str.strip()
    df = df.drop(columns=[c for c in ["inaccessible_pole1", "railroad_pole1", "inaccessible_pole2",
                                      "railroad_pole2"] if c in df.columns])

    for pole_col, suffix in [(p1, "pole1"), (p2, "pole2")]:
        df = df.merge(
            lookup.rename(columns={"client_pole_id": pole_col, "inaccessible_flag": f"inaccessible_{suffix}",
                                   "railroad_flag": f"railroad_{suffix}"}),
            on=pole_col, how="left",
        )

    df["access_missing_both_poles"] = df["inaccessible_pole1"].isna() & df["inaccessible_pole2"].isna()
    df["inaccessible_raw"] = (df["inaccessible_pole1"].fillna(True).astype(bool)
                              | df["inaccessible_pole2"].fillna(True).astype(bool)).astype(int)
    df["railroad_span"] = (df["railroad_pole1"].fillna(True).astype(bool)
                           | df["railroad_pole2"].fillna(True).astype(bool)).astype(int)

    # Unknown on both poles -> fall back to truck access (truck_access==0 => inaccessible)
    df["inaccessible"] = df["inaccessible_raw"]
    df["accessibility_fill_from_truck_access"] = False
    if "truck_access" in df.columns:
        # Notebook behaviour: inaccessible = not bool(truck_access), so the -1 "unknown" code becomes
        # ACCESSIBLE. Kept as the default to reproduce the saved splits; review whether that is intended.
        truck_known = df["truck_access"].isin([0, 1]) | unknown_truck_as_accessible
        fill = df["access_missing_both_poles"] & truck_known
        df.loc[fill, "inaccessible"] = (df.loc[fill, "truck_access"] == 0).astype(int)
        df.loc[fill, "accessibility_fill_from_truck_access"] = True
    print("Mean manhours by accessibility:\n", df.groupby("inaccessible")[TARGET].mean())
    return df


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Severity-colour intrusion bands + derived ratios (identical to the notebook)."""
    f = frame.copy()
    eps = 1e-6
    f["heavy_contact_vol"] = f["bay_red_volume_(cubic_ft)"] + f["bay_orange_volume_(cubic_ft)"]
    f["heavy_contact_count"] = f["bay_red_count"] + f["bay_orange_count"]
    f["heavy_contact_flag"] = (f["heavy_contact_vol"] > 0).astype(int)
    f["moderate_contact_vol"] = f["bay_yellow_volume_(cubic_ft)"] + f["bay_green_volume_(cubic_ft)"]
    f["moderate_contact_count"] = f["bay_yellow_count"] + f["bay_green_count"]
    f["moderate_contact_flag"] = (f["moderate_contact_vol"] > 0).astype(int)
    f["low_contact_vol"] = f["bay_blue_volume_(cubic_ft)"]
    f["low_contact_count"] = f["bay_blue_count"]
    f["low_contact_flag"] = (f["low_contact_vol"] > 0).astype(int)
    f["side_intrusion_vol"] = (f["bay_total_intrusion_volume_(cubic_ft)"]
                               - f["bay_os_volume_(cubic_ft)"]
                               - f["bay_purple_volume_(cubic_ft)"]).clip(lower=0)
    total = f["bay_total_intrusion_volume_(cubic_ft)"] + eps
    f["heavy_contact_ratio"] = f["heavy_contact_vol"] / total
    f["moder_contact_ratio"] = f["moderate_contact_vol"] / total
    f["low_contact_ratio"] = f["low_contact_vol"] / total
    f["trim_vol_by_span"] = f["side_intrusion_vol"] * f["bay_total_span_length_(ft)"]
    f["heavy_contact_cond"] = f["heavy_contact_vol"] / (f["num_conductors"] + eps)
    f["trim_vol_by_length"] = (f["side_intrusion_vol"] / (f["bay_length_(ft)"] + eps)) * f["bay_total_span_length_(ft)"]
    f["intrusion_per_ft"] = f["bay_total_intrusion_volume_(cubic_ft)"] / (f["bay_length_(ft)"] + eps)
    lin = sum(f[c].fillna(0) if c in f.columns else 0
              for c in ["bay_l&r_red_(ft)", "bay_l&r_orange_(ft)", "bay_l&r_yellow_(ft)", "bay_l&r_green_(ft)"])
    f["heavy_lin_to_ttl_lin"] = lin / (f["bay_length_(ft)"] + eps)
    f["railroad_terrain_interaction"] = f["railroad_span"] * f["bay_total_intrusion_volume_(cubic_ft)"]
    return f


ENGINEERED_FEATURES = [
    "side_intrusion_vol", "heavy_contact_vol", "heavy_contact_count", "heavy_contact_flag",
    "heavy_contact_ratio", "low_contact_vol", "moderate_contact_vol", "low_contact_count",
    "moderate_contact_count", "low_contact_flag", "moderate_contact_flag", "avg_stem_size",
    "intrusion_per_ft", "overhang_flag", "undergrowth_intensity", "max_voltage_kV",
    "num_circuits", "num_conductors", "min_distance_to_ccl", "railroad",
    "num_phases_cegis", "heavy_lin_to_ttl_lin", "moder_contact_ratio", "low_contact_ratio",
    "railroad_terrain_interaction", "trim_vol_by_span", "trim_vol_by_length", "heavy_contact_cond",
    "inaccessible",
]
ID_COLUMNS = ["bay_site1_id", "bay_site2_id", "bay_publication_id", "bay_ma_number", "bay_unique"]


def build_model_frame(df: pd.DataFrame):
    """Fill, engineer, flag non-veg, drop missing targets / non-veg spans. Returns (df, feature_cols)."""
    raw = [c for c in df.columns if c.startswith("bay_")
           and df[c].dtype.kind in "if" and c not in ID_COLUMNS]
    df[TARGET] = df[TARGET].fillna(-1)
    df[raw] = df[raw].fillna(0)            # constant fill: no information from other rows
    df = engineer_features(df)

    df["nonveg_span"] = (df["bay_total_intrusion_volume_(cubic_ft)"].isna()
                         | df["bay_total_intrusion_volume_(cubic_ft)"].eq(0)).astype(int)
    df = df.drop(columns=[c for c in ["index"] if c in df.columns])
    df = df[df[TARGET] != -1].copy()

    # Analysis-only columns (post-work counts) - never model inputs.
    veg_cols = [c for c in ["total_trims", "total_removals", "brush_removals", "brush_trims"] if c in df.columns]
    if veg_cols:
        df["total_veg_work"] = df[veg_cols].sum(axis=1, min_count=1)
        has_work = df["total_veg_work"].fillna(0) > 0
        df.loc[has_work, "veg_tier"] = pd.qcut(
            df.loc[has_work, "total_veg_work"].rank(method="first"), q=[0, 0.25, 0.75, 1.0],
            labels=["Light", "Medium", "Heavy"]).astype(str)

    print(f"Non-veg spans removed: {int(df['nonveg_span'].sum()):,}")
    df = df[df["nonveg_span"] == 0].copy()

    engineered = [c for c in ENGINEERED_FEATURES if c in df.columns]
    feature_cols = sorted(set(engineered + raw))
    print(f"Candidate features: {len(feature_cols)} | spans: {len(df):,}")
    return df, feature_cols


def bootstrap_ci(series, confidence_level=0.95, n_resamples=5000, random_state=RANDOM_STATE):
    values = pd.Series(series).dropna().to_numpy()
    if len(values) < 2:
        return np.nan, np.nan
    res = bootstrap((values,), np.mean, confidence_level=confidence_level, n_resamples=n_resamples,
                    random_state=random_state, method="percentile")
    return res.confidence_interval.low, res.confidence_interval.high


def segment_summary(frame: pd.DataFrame, by=("inaccessible", "veg_tier")) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(list(by)):
        lo, hi = bootstrap_ci(g[TARGET], n_resamples=1000)
        rows.append({**dict(zip(by, key)), "count": len(g), "mean": g[TARGET].mean(),
                     "median": g[TARGET].median(), "p10": g[TARGET].quantile(0.1),
                     "p90": g[TARGET].quantile(0.9), "mean_ci_lower": lo, "mean_ci_upper": hi})
    return pd.DataFrame(rows)


# =============================================================================
# 3. FEEDER-GROUPED SPLITS (recent-window holdout)
# =============================================================================
@dataclass
class SplitConfig:
    holdout_target_share: float = 0.20
    holdout_share_tolerance: float = 0.03
    dev_temp_share: float = 0.30
    min_holdout_spans: int = 500
    min_holdout_feeders: int = 20
    random_state: int = RANDOM_STATE


def split_paths(split_dir: str, split_name: str) -> Dict[str, str]:
    return {part: os.path.join(split_dir, f"{part}_{split_name}.csv") for part in ["X", "y", "meta"]}


def splits_available(split_dir: str) -> bool:
    return all(os.path.exists(p) for s in SPLIT_NAMES for p in split_paths(split_dir, s).values())


def save_splits(splits: dict, split_dir: str):
    os.makedirs(split_dir, exist_ok=True)
    for name, data in splits.items():
        paths = split_paths(split_dir, name)
        data["X"].to_csv(paths["X"], index=True)
        data["y"].to_frame(TARGET).to_csv(paths["y"], index=True)
        data["meta"].to_csv(paths["meta"], index=True)
    print(f"Saved train/test/valid/holdout splits to {split_dir}")


def load_splits(split_dir: str) -> dict:
    loaded = {}
    for name in SPLIT_NAMES:
        paths = split_paths(split_dir, name)
        X = pd.read_csv(paths["X"], index_col=0)
        y = pd.read_csv(paths["y"], index_col=0)[TARGET].astype(float)
        meta = pd.read_csv(paths["meta"], index_col=0)
        for frame in (X, y, meta):
            frame.index.name = "bay_unique"
        meta["date_only"] = pd.to_datetime(meta["date_only"], errors="coerce")
        loaded[name] = {"X": X, "y": y, "meta": meta}
    print(f"Loaded saved splits from {split_dir}")
    return loaded


def make_feeder_splits(df: pd.DataFrame, feature_cols: List[str], cfg: SplitConfig = SplitConfig()) -> dict:
    """Holdout = feeders whose whole (fully dated) history lies in the latest 1/2/3 months.
    Remaining feeders -> 70/15/15 train/valid/test. No feeder in more than one split."""
    data = df.copy()
    if data.index.name != "bay_unique":
        data = data.set_index("bay_unique", drop=False)
    data.index.name = "bay_unique"
    if data.index.duplicated().any():
        raise ValueError("bay_unique is not unique.")

    data["date_only"] = pd.to_datetime(data["date_only"], errors="coerce").dt.normalize()
    n_missing_feeder = int(data["feeder"].isna().sum())
    data = data[data["feeder"].notna()].copy()
    print(f"Removed {n_missing_feeder:,} spans with missing feeder.")
    data["missing_complete_date_flag"] = data["date_only"].isna().astype(int)

    latest = data["date_only"].max()
    month_start = latest.to_period("M").start_time
    windows = [
        ("Latest calendar month", month_start),
        ("Latest two calendar months", month_start - pd.DateOffset(months=1)),
        ("Latest three calendar months", month_start - pd.DateOffset(months=3)),
    ]

    total = len(data)
    target_spans = int(round(total * cfg.holdout_target_share))
    min_spans = max(cfg.min_holdout_spans,
                    int(round(total * (cfg.holdout_target_share - cfg.holdout_share_tolerance))))

    fsum = data.groupby("feeder").agg(
        feeder_spans=("date_only", "size"),
        missing=("date_only", lambda s: s.isna().sum()),
        first=("date_only", "min"),
        last=("date_only", "max"),
    ).reset_index()

    def eligible(start):
        e = fsum[(fsum["missing"] == 0) & (fsum["first"] >= start) & (fsum["last"] <= latest)]
        return e.sort_values(["feeder_spans", "last", "first", "feeder"],
                             ascending=[False, False, False, True]).reset_index(drop=True)

    def near_target(e):
        if e.empty:
            return np.array([], dtype=object)
        cum = np.cumsum(e["feeder_spans"].to_numpy())
        return e.loc[: int(np.argmin(np.abs(cum - target_spans))), "feeder"].to_numpy()

    chosen, chosen_window = None, None
    for label, start in windows:
        cand = near_target(eligible(start))
        n_spans = int(data["feeder"].isin(cand).sum())
        print(f"{label}: {len(cand)} feeders, {n_spans:,} spans ({n_spans / total:.1%})")
        if n_spans >= min_spans and len(cand) >= cfg.min_holdout_feeders:
            chosen, chosen_window = cand, (label, start)
            break
    if chosen is None:
        label, start = windows[-1]
        e = eligible(start)
        if e.empty:
            raise ValueError("No fully dated feeders inside the latest three months - cannot build holdout.")
        chosen, chosen_window = near_target(e), (label, start)
        print("WARNING: holdout below preferred size; using largest feasible three-month group.")

    holdout_mask = data["feeder"].isin(chosen)
    dev_feeders = data.loc[~holdout_mask, "feeder"].unique()
    f_train, f_temp = train_test_split(dev_feeders, test_size=cfg.dev_temp_share, random_state=cfg.random_state)
    f_valid, f_test = train_test_split(f_temp, test_size=0.5, random_state=cfg.random_state)

    masks = {"train": data["feeder"].isin(f_train), "test": data["feeder"].isin(f_test),
             "valid": data["feeder"].isin(f_valid), "holdout": holdout_mask}
    sets = {k: set(data.loc[m, "feeder"]) for k, m in masks.items()}
    for a, b in combinations(sets, 2):
        if sets[a] & sets[b]:
            raise ValueError(f"Feeder leakage between {a} and {b}")
    if not sum(m.astype(int) for m in masks.values()).eq(1).all():
        raise ValueError("Rows assigned to zero or multiple splits.")
    if data.loc[holdout_mask, "date_only"].isna().any():
        raise ValueError("Holdout contains missing dates.")

    meta_cols = ["feeder", "date_only", "missing_complete_date_flag"] + [
        c for c in ["inaccessible", "veg_tier"] if c in data.columns]
    feats = [c for c in feature_cols if c in data.columns]
    splits = {k: {"X": data.loc[m, feats].copy(), "y": data.loc[m, TARGET].astype(float).copy(),
                  "meta": data.loc[m, meta_cols].copy()} for k, m in masks.items()}
    print(f"Holdout window: {chosen_window[0]} ({chosen_window[1].date()} - {latest.date()})")
    return splits


def split_summary(splits: dict) -> pd.DataFrame:
    total = sum(len(s["X"]) for s in splits.values())
    return pd.DataFrame([{
        "split": k, "spans": len(s["X"]), "share": len(s["X"]) / total,
        "feeders": s["meta"]["feeder"].nunique(),
        "missing_dates": int(s["meta"]["date_only"].isna().sum()),
        "min_date": s["meta"]["date_only"].min(), "max_date": s["meta"]["date_only"].max(),
    } for k, s in splits.items()]).set_index("split")


# =============================================================================
# 4. LEGACY COLUMN CLEANUP
# =============================================================================
def drop_legacy_tier_features(splits: dict) -> dict:
    """v3 removes the KMeans / hierarchical tier features entirely (they were also fit pre-split)."""
    for s in splits:
        splits[s]["X"] = splits[s]["X"].drop(columns=[c for c in LEGACY_TIER_FEATURE_COLS
                                                      if c in splits[s]["X"].columns])
    return splits


# =============================================================================
# 5. ACCESSIBILITY ROUTING + SHARED-PARAM MODEL PAIRS (Section 6 baseline/challenger)
# =============================================================================
def build_accessibility_lookup(splits: dict) -> pd.Series:
    lookup = pd.concat([splits[s]["meta"]["inaccessible"] for s in SPLIT_NAMES])
    if lookup.index.duplicated().any():
        raise ValueError("Accessibility lookup has duplicated bay_unique values.")
    lookup = pd.to_numeric(lookup, errors="coerce").astype("Int64")
    if (~lookup.dropna().isin([0, 1])).any():
        raise ValueError("Accessibility flags must be 0/1.")
    return lookup


def get_accessibility_flags(lookup: pd.Series, index) -> pd.Series:
    index = pd.Index(index)
    missing = index.difference(lookup.index)
    if len(missing):
        raise KeyError(f"{len(missing)} rows missing from accessibility lookup, e.g. {missing[:5].tolist()}")
    flags = lookup.reindex(index)
    if flags.isna().any():
        raise ValueError(f"{int(flags.isna().sum())} rows have missing accessibility flags.")
    return flags.astype(int)


def align_target(y, X, name="y") -> pd.Series:
    if isinstance(y, pd.Series) and X.index.isin(y.index).all():
        out = y.reindex(X.index)
    elif len(y) == len(X):
        out = pd.Series(np.asarray(y), index=X.index, name=name)
    else:
        raise ValueError(f"Unable to align {name}.")
    if out.isna().any():
        raise ValueError(f"{name} has missing values after alignment.")
    return out.astype(float)


def fit_accessibility_pair(X, y, flags: pd.Series, params: dict, cat_features=None, label=""):
    """Section-6 style pair with one shared param set (baseline / road challenger)."""
    models = {}
    for seg, flag in SEGMENTS.items():
        m = flags.reindex(X.index).eq(flag).to_numpy()
        model = CatBoostRegressor(**params)
        model.fit(X[m], y[m], cat_features=cat_features, verbose=False)
        models[seg] = model
        print(f"{label}{seg}: fit on {int(m.sum()):,} spans")
    return models


def predict_accessibility_pair(models: dict, X, flags: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=X.index, dtype=float)
    f = flags.reindex(X.index)
    for seg, flag in SEGMENTS.items():
        idx = f.index[f.eq(flag).to_numpy()]
        if len(idx):
            out.loc[idx] = models[seg].predict(X.loc[idx])
    if out.isna().any():
        raise ValueError("Some rows were not routed to a model.")
    return out.clip(lower=0)


def calculate_metrics(name, actual, predicted) -> dict:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    r = p - a
    return {"Model": name, "Count": len(a), "MAE": mean_absolute_error(a, p),
            "Median_AE": median_absolute_error(a, p), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "R2": r2_score(a, p), "Bias": r.mean(), "P90_AE": np.quantile(np.abs(r), 0.9),
            "Underprediction_Rate": float(np.mean(p < a))}


# =============================================================================
# 6. ROAD / ALLEY / LAND-COVER / AREA FEATURE JOIN
# =============================================================================
BOOLEAN_MAP = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0, "n": 0.0, "1": 1.0, "0": 0.0}


def clean_bay_key(series: pd.Series) -> pd.Series:
    s = (series.astype("string").str.strip()
         .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA, "null": pd.NA}))
    return s.str.replace(r"\.0$", "", regex=True)


def resolve_road_feature_csv(extra_candidates: Sequence[str] = ()) -> Path:
    here = Path.cwd().resolve()
    cands = [Path(c) for c in extra_candidates] + [
        here / "../input_data/Accessibility/bay_geometry_df.csv",
        here / "input_data/Accessibility/bay_geometry_df.csv",
        Path("/mnt/projects/VEG_Contractors/2026unitization/2026unitization/input_data/Accessibility/bay_geometry_df.csv"),
    ] + [p / "input_data/Accessibility/bay_geometry_df.csv" for p in here.parents]
    for c in cands:
        if c.resolve().exists():
            return c.resolve()
    raise FileNotFoundError("bay_geometry_df.csv not found. Checked:\n" + "\n".join(map(str, cands)))


def load_road_features(path, numeric_requested=REQUESTED_NUMERIC_ROAD_FEATURES,
                       categorical_requested=REQUESTED_CATEGORICAL_ROAD_FEATURES):
    """One aggregated road-feature row per bay (numeric mean, categorical deterministic mode)."""
    src = pd.read_csv(path, low_memory=False)
    src.columns = src.columns.astype(str).str.strip()
    src = src.loc[:, ~src.columns.duplicated()].copy()
    src["bay"] = clean_bay_key(src["bay"])
    src = src[src["bay"].notna()].copy()

    numeric = [c for c in numeric_requested if c in src.columns]
    categorical = [c for c in categorical_requested if c in src.columns]
    missing = sorted(set(numeric_requested + categorical_requested) - set(numeric + categorical))
    print(f"Road features available: {len(numeric)} numeric, {len(categorical)} categorical")
    if missing:
        print(f"Optional road fields not found ({len(missing)}): {missing}")

    for c in numeric:
        num = pd.to_numeric(src[c], errors="coerce")
        src[c] = num.fillna(src[c].astype("string").str.strip().str.lower().map(BOOLEAN_MAP))
    for c in categorical:
        src[c] = (src[c].astype("string").str.strip()
                  .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).fillna("Unknown"))

    def mode(s):
        v = s.dropna().astype(str)
        return "Unknown" if v.empty else sorted(v.mode().tolist())[0]

    parts = []
    if numeric:
        parts.append(src.groupby("bay")[numeric].mean())
    if categorical:
        parts.append(src.groupby("bay")[categorical].agg(mode))
    by_bay = pd.concat(parts, axis=1)
    by_bay.index = by_bay.index.astype("string")
    return by_bay, numeric, categorical


def lookup_road_features(X: pd.DataFrame, bay_key: pd.Series, road_by_bay: pd.DataFrame) -> pd.DataFrame:
    bays = bay_key.reindex(X.index)
    if bays.isna().any():
        raise ValueError(f"{int(bays.isna().sum())} rows have no bay join key.")
    frame = road_by_bay.reindex(bays.to_numpy()).copy()
    frame.index = X.index
    frame["road_feature_source_match"] = frame.notna().any(axis=1).astype(float)
    frame["road_feature_source_missing"] = 1.0 - frame["road_feature_source_match"]
    return frame


def fit_road_transforms(road_train: pd.DataFrame, numeric: List[str], categorical: List[str],
                        min_category_count: int = 20) -> dict:
    """Category vocabularies and numeric medians from TRAIN only."""
    vocab = {}
    for c in categorical:
        counts = road_train[c].astype("string").fillna("Unknown").value_counts()
        vocab[c] = set(counts[counts >= min_category_count].index.astype(str)) | {"Unknown", "Other"}
    num_cols = numeric + ["road_feature_source_match", "road_feature_source_missing"]
    med = road_train[num_cols].median(numeric_only=True).reindex(num_cols).fillna(0.0)
    med["road_feature_source_match"], med["road_feature_source_missing"] = 0.0, 1.0
    return {"vocab": vocab, "numeric_cols": num_cols, "medians": med, "categorical": categorical}


def apply_road_transforms(frame: pd.DataFrame, t: dict) -> pd.DataFrame:
    frame = frame.copy()
    for c in t["categorical"]:
        v = frame[c].astype("string").fillna("Unknown").astype(str)
        frame[c] = np.where(v.isin(t["vocab"][c]), v, "Other").astype(str)
    frame[t["numeric_cols"]] = (frame[t["numeric_cols"]].apply(pd.to_numeric, errors="coerce")
                                .fillna(t["medians"]).astype(float))
    return frame


def build_road_matrix(X_base: pd.DataFrame, road_frame: pd.DataFrame, df: pd.DataFrame,
                      road_cols: List[str]) -> pd.DataFrame:
    overlap = set(X_base.columns) & set(road_cols)
    if overlap:
        raise ValueError(f"Base matrix already has road columns: {sorted(overlap)[:10]}")
    out = pd.concat([X_base, road_frame[road_cols]], axis=1)
    out["heavy_contact_conductor"] = (
        (df.loc[X_base.index, "heavy_contact_vol"] + df.loc[X_base.index, "moderate_contact_vol"])
        * df.loc[X_base.index, "num_conductors"]
    ).to_numpy()
    if not out.index.equals(X_base.index):
        raise ValueError("Road join changed the index.")
    return out


def tune_shared_road_params(X, y, feeders, cat_cols, n_trials, n_folds=5, max_iter_range=(500, 1800),
                            random_state=RANDOM_STATE, show_progress=True):
    """Section 6 shared-param study (one param set for both regimes). Early stopping on a feeder
    slice of the training fold, never on the scored fold."""
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=min(n_folds, feeders.nunique()))

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", *max_iter_range, step=100),
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.5),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "loss_function": "MAE", "eval_metric": "MAE", **CB_FIXED,
        }
        maes, iters = [], []
        for k, (tr, ev) in enumerate(cv.split(X, y, groups=feeders), start=1):
            rng = np.random.default_rng(random_state + k)
            tr_feeders = feeders.iloc[tr]
            uniq = tr_feeders.unique()
            es_f = rng.choice(uniq, size=max(1, int(0.15 * len(uniq))), replace=False)
            es = tr_feeders.isin(es_f).to_numpy()
            model = CatBoostRegressor(**params)
            model.fit(X.iloc[tr[~es]], y.iloc[tr[~es]], cat_features=cat_cols,
                      eval_set=(X.iloc[tr[es]], y.iloc[tr[es]]), early_stopping_rounds=100,
                      use_best_model=True, verbose=False)
            maes.append(mean_absolute_error(y.iloc[ev], np.clip(model.predict(X.iloc[ev]), 0, None)))
            iters.append(model.get_best_iteration() + 1)
            trial.report(np.mean(maes), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("median_best_iteration", int(np.median(iters)))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=random_state),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress, gc_after_trial=True)
    params = dict(study.best_params)
    params["iterations"] = max(100, study.best_trial.user_attrs.get("median_best_iteration", params["iterations"]))
    print(f"Shared road tuning: {(time.time() - t0) / 60:.1f} min, best grouped-CV MAE {study.best_value:.4f}")
    return params, study


# =============================================================================
# 7. AREA CLASS + FEATURE GROUPS + FULL-FEATURE SPLIT IO
# =============================================================================
def to_area_class(values, column: str, value_map: Optional[dict] = None) -> pd.Series:
    value_map = AREA_VALUE_MAP if value_map is None else value_map
    text = pd.Series(values).astype(str).str.strip()
    explicit = text.map(value_map.get(column, {}))
    low = text.str.lower()
    auto = pd.Series(np.select([low.str.contains("suburb"), low.str.contains("urban"), low.str.contains("rural")],
                               ["Suburban", "Urban", "Rural"], default="Unmapped"), index=text.index)
    return explicit.fillna(auto)


def build_feature_groups(full_columns: List[str], road_cols: List[str]) -> dict:
    alley = [c for c in road_cols if "alley" in c.lower()]
    lc_num = [c for c in LAND_COVER_NUMERIC_FEATURES if c in road_cols]
    lc_cat = [c for c in LAND_COVER_CATEGORICAL_FEATURES if c in road_cols]
    road_core = [c for c in road_cols if c not in set(alley + lc_num + lc_cat + AREA_CANDIDATE_COLS)]
    return {
        "base_span": [c for c in full_columns if c not in set(road_cols)],
        "road_core": road_core,
        "alley": alley,
        "land_cover_numeric": lc_num,
        "land_cover_categorical": lc_cat,
        "area_candidates": [c for c in AREA_CANDIDATE_COLS if c in full_columns],
    }


def build_meta(X: pd.DataFrame, df: pd.DataFrame, flags: pd.Series, value_map=None) -> pd.DataFrame:
    meta = pd.DataFrame(index=X.index)
    meta["feeder"] = df.loc[X.index, "feeder"].to_numpy()
    meta["inaccessible"] = flags.reindex(X.index).to_numpy()
    for c in ["impactful_issues", "veg_tier", "bay"]:
        if c in df.columns:
            meta[c] = df.loc[X.index, c].to_numpy()
    for c in AREA_CANDIDATE_COLS:
        if c in X.columns:
            meta[f"area_class__{c}"] = to_area_class(X[c], c, value_map).to_numpy()
    return meta


def assert_no_leaky_columns(columns: Iterable[str]):
    leaky = [c for c in columns if any(p in c.lower() for p in LEAKY_NAME_PATTERNS)]
    if leaky:
        raise ValueError(f"Post-work / target-derived columns in features: {leaky}")


def save_full_splits(full_x, full_y, full_meta, out_dir, manifest: dict):
    os.makedirs(out_dir, exist_ok=True)
    for s in full_x:
        full_x[s].to_csv(os.path.join(out_dir, f"X_{s}.csv"), index=True)
        full_y[s].to_frame(TARGET).to_csv(os.path.join(out_dir, f"y_{s}.csv"), index=True)
        full_meta[s].to_csv(os.path.join(out_dir, f"meta_{s}.csv"), index=True)
    write_json(manifest, os.path.join(out_dir, "feature_manifest.json"))
    print(f"Saved full-feature splits ({full_x['train'].shape[1]} features) to {out_dir}")


def load_full_splits(out_dir):
    manifest = read_json(os.path.join(out_dir, "feature_manifest.json"))
    X, y, meta = {}, {}, {}
    for s in ["train", "valid", "test", "holdout"]:
        xs = pd.read_csv(os.path.join(out_dir, f"X_{s}.csv"), index_col=0)
        for c in manifest["categorical_columns"]:
            xs[c] = xs[c].astype(str)
        X[s] = xs[manifest["feature_columns"]]
        y[s] = pd.read_csv(os.path.join(out_dir, f"y_{s}.csv"), index_col=0)[TARGET].astype(float)
        meta[s] = pd.read_csv(os.path.join(out_dir, f"meta_{s}.csv"), index_col=0)
    return X, y, meta, manifest


# =============================================================================
# 8. MODEL UNITS + DEVELOPMENT HARNESS (feeder-grouped CV on Train + Validation)
# =============================================================================
@dataclass
class ModelUnit:
    """One trained model. `areas=None` means all area classes.

    families: loss labels Optuna may choose from ("MAE", "RMSE", "Huber", "Tweedie", "Q0.55", ...).
    conditional_quantiles: alphas tried ONLY if CV still shows underprediction after tuning `families`.
    train_scope: "unit" (fit on the unit's rows), "segment" (fit on the whole accessibility regime,
                 score on the unit's rows) or "auto" (unit unless it has < cfg["min_unit_spans"] dev rows).
    """
    name: str
    inaccessible: int
    areas: Optional[List[str]] = None
    families: List[str] = field(default_factory=lambda: ["MAE", "Huber", "Tweedie"])
    conditional_quantiles: List[float] = field(default_factory=list)
    train_scope: str = "auto"

    @property
    def segment(self) -> str:
        return "Inaccessible" if self.inaccessible == 1 else "Accessible"


def default_model_units() -> List[ModelUnit]:
    return [
        ModelUnit("Accessible", 0, None, ["MAE", "Huber", "Tweedie"]),
        ModelUnit("Inaccessible-Suburban", 1, ["Suburban"], ["Huber", "Tweedie", "Q0.55", "Q0.60"]),
        ModelUnit("Inaccessible-Rural", 1, ["Rural"], ["MAE", "Huber", "Tweedie"], [0.55, 0.60]),
        # TODO(confirm): urban inaccessible was not specified - treated like rural. Unmapped area rows
        # land here too so every inaccessible span has exactly one model.
        ModelUnit("Inaccessible-Urban", 1, ["Urban", "Unmapped"], ["MAE", "Huber", "Tweedie"], [0.55, 0.60]),
    ]


def segment_units() -> List[ModelUnit]:
    """Plain accessible / inaccessible pair (used before the area column is chosen)."""
    return [ModelUnit(seg, flag, None, ["RMSE"], train_scope="unit") for seg, flag in SEGMENTS.items()]


def unit_mask(meta: pd.DataFrame, unit: ModelUnit, area_class_col: Optional[str]) -> np.ndarray:
    m = meta["inaccessible"].eq(unit.inaccessible).to_numpy()
    if unit.areas is not None:
        m = m & meta[area_class_col].isin(unit.areas).to_numpy()
    return m


def scope_mask(meta, unit, area_class_col, scope):
    return unit_mask(meta, unit, area_class_col) if scope == "unit" else meta["inaccessible"].eq(unit.inaccessible).to_numpy()


@dataclass
class DevContext:
    full_x: dict
    full_y: dict
    full_meta: dict
    categorical_pool: List[str]
    cfg: dict
    random_state: int = RANDOM_STATE
    area_class_col: Optional[str] = None
    X_dev: pd.DataFrame = field(init=False)
    y_dev: pd.Series = field(init=False)
    meta_dev: pd.DataFrame = field(init=False)
    folds: list = field(init=False)

    def __post_init__(self):
        self.X_dev = pd.concat([self.full_x["train"], self.full_x["valid"]])
        self.y_dev = pd.concat([self.full_y["train"], self.full_y["valid"]]).reindex(self.X_dev.index)
        self.meta_dev = pd.concat([self.full_meta["train"], self.full_meta["valid"]]).reindex(self.X_dev.index)
        assert not self.X_dev.index.duplicated().any(), "Duplicate bay_unique in Train+Validation"
        assert self.y_dev.notna().all(), "Missing targets in Train+Validation"
        for other in ("test", "holdout"):
            assert set(self.meta_dev["feeder"]).isdisjoint(self.full_meta[other]["feeder"]), f"dev/{other} feeder overlap"
        assert_no_leaky_columns(self.X_dev.columns)
        self.folds = make_feeder_folds(self.X_dev.index, self.meta_dev["feeder"].to_numpy(),
                                       self.cfg["n_folds"], self.cfg["es_feeder_share"], self.random_state)

    def rows(self, unit: ModelUnit, scope="unit") -> pd.Index:
        return self.meta_dev.index[scope_mask(self.meta_dev, unit, self.area_class_col, scope)]


def resolve_units(ctx: DevContext, units: List[ModelUnit], verbose=True) -> List[ModelUnit]:
    """Drop empty units, resolve train_scope='auto', and check every dev/test/holdout row has one unit."""
    out = []
    for u in units:
        n = int(unit_mask(ctx.meta_dev, u, ctx.area_class_col).sum())
        if n == 0:
            print(f"  unit {u.name}: no dev rows - skipped")
            continue
        u = copy_unit(u)
        if u.train_scope == "auto":
            u.train_scope = "unit" if n >= ctx.cfg["min_unit_spans"] else "segment"
        out.append(u)
        if verbose:
            print(f"  unit {u.name:24s} dev rows={n:6,} train_scope={u.train_scope:8s} families={u.families}"
                  + (f" conditional quantiles={u.conditional_quantiles}" if u.conditional_quantiles else ""))
    for split, meta in {"dev": ctx.meta_dev, **{s: ctx.full_meta[s] for s in ("test", "holdout")}}.items():
        cover = sum(unit_mask(meta, u, ctx.area_class_col).astype(int) for u in out)
        if not (cover == 1).all():
            raise ValueError(f"{split}: {int((cover != 1).sum())} rows are covered by zero or several units - "
                             "adjust unit areas (e.g. add 'Unmapped' somewhere).")
    return out


def copy_unit(u: ModelUnit) -> ModelUnit:
    return ModelUnit(u.name, u.inaccessible, None if u.areas is None else list(u.areas),
                     list(u.families), list(u.conditional_quantiles), u.train_scope)


def make_feeder_folds(index, feeders, n_splits, es_share, seed):
    """Shuffled, span-balanced feeder folds; each training fold also yields an early-stopping feeder slice."""
    rng = np.random.default_rng(seed)
    fser = pd.Series(feeders, index=index)
    sizes = fser.value_counts().sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")
    load, fold_of = np.zeros(n_splits), {}
    for feeder, size in sizes.items():
        k = int(np.argmin(load + rng.random(n_splits) * 1e-6))
        fold_of[feeder] = k
        load[k] += size
    row_fold = fser.map(fold_of).to_numpy()
    folds = []
    for k in range(n_splits):
        ev, tr = index[row_fold == k], index[row_fold != k]
        tr_f = fser.loc[tr]
        uniq = tr_f.unique()
        es_f = set(rng.choice(uniq, size=max(1, int(round(len(uniq) * es_share))), replace=False))
        in_es = tr_f.isin(es_f).to_numpy()
        folds.append({"fit": tr[~in_es], "es": tr[in_es], "eval": ev})
    return folds


# ---------------------------------------------------------------------------
# Loss families
# ---------------------------------------------------------------------------
def loss_label_to_catboost(label: str, trial=None, fixed: Optional[dict] = None) -> str:
    """Map a family label to a CatBoost loss string (suggesting shape params when a trial is given)."""
    fixed = fixed or {}
    if label in ("MAE", "RMSE"):
        return label
    if label == "Huber":
        delta = trial.suggest_float("huber_delta", 0.5, 5.0, log=True) if trial else fixed.get("huber_delta", 2.0)
        return f"Huber:delta={delta:.3f}"
    if label == "Tweedie":
        power = trial.suggest_float("tweedie_power", 1.1, 1.9) if trial else fixed.get("tweedie_power", 1.5)
        return f"Tweedie:variance_power={power:.3f}"
    if label.startswith("Q"):
        return f"Quantile:alpha={float(label[1:]):g}"
    raise ValueError(label)


def quantile_label(alpha: float) -> str:
    return f"Q{alpha:.2f}"


def make_loss_builder(families: List[str]):
    families = list(families)

    def builder(trial):
        return loss_label_to_catboost(trial.suggest_categorical("loss_family", families), trial)
    return builder


def loss_name(params) -> str:
    return str(params.get("loss_function", "RMSE"))


# ---------------------------------------------------------------------------
# CatBoost fit / predict
# ---------------------------------------------------------------------------
def make_pool(ctx: DevContext, X, y, features, weight=None):
    return Pool(X[features], None if y is None else np.asarray(y, float),
                cat_features=[c for c in ctx.categorical_pool if c in features],
                weight=None if weight is None else np.asarray(weight, float))


def fit_cb(ctx: DevContext, params, features, X_fit, y_fit, X_es=None, y_es=None, iterations=None, weight=None):
    p = {**CB_FIXED, "random_seed": ctx.random_state, **params}
    p["iterations"] = int(iterations or p.get("iterations", ctx.cfg["max_iterations"]))
    model = CatBoostRegressor(**p)
    train_pool = make_pool(ctx, X_fit, y_fit, features, weight)
    if X_es is not None and len(X_es):
        # Early-stopping pool is UNWEIGHTED: stopping is judged on the plain loss.
        model.fit(train_pool, eval_set=make_pool(ctx, X_es, y_es, features),
                  early_stopping_rounds=ctx.cfg["early_stopping_rounds"], use_best_model=True)
    else:
        model.fit(train_pool)
    return model


def predict_cb(ctx: DevContext, model, params, X, features) -> np.ndarray:
    pool = make_pool(ctx, X, None, features)
    if loss_name(params).startswith(("Tweedie", "Poisson")):
        raw = model.predict(pool, prediction_type="Exponent")
    else:
        raw = model.predict(pool)
    return np.clip(np.asarray(raw, float), 0, None)


def tier_index(values, cuts) -> np.ndarray:
    """Right-inclusive, identical to pd.cut(bins=[-inf, *cuts, inf])."""
    return np.searchsorted(np.asarray(cuts, float), np.asarray(values, float), side="left")


def feeder_wape(actual, pred, feeders) -> float:
    fr = pd.DataFrame({"a": actual, "p": pred, "f": feeders}).groupby("f")[["a", "p"]].sum()
    return float((fr["p"] - fr["a"]).abs().sum() / max(fr["a"].sum(), 1e-9))


def cv_unit(ctx: DevContext, unit: ModelUnit, features, params, *, weight_fn=None, importance=False,
            fold_callback=None, quantile_alphas=None) -> dict:
    """Feeder-grouped CV for one model unit. Fit rows follow unit.train_scope; eval rows are the unit's rows.

    weight_fn(index) -> per-row weights for the FIT rows (None = unweighted).
    """
    unit_rows = ctx.rows(unit, "unit")
    fit_scope = ctx.rows(unit, unit.train_scope)
    oof = pd.Series(np.nan, index=unit_rows, dtype=float)
    oof_q = None if quantile_alphas is None else pd.DataFrame(np.nan, index=unit_rows, columns=list(quantile_alphas))
    maes, wapes, iters, imps = [], [], [], []
    for k, fold in enumerate(ctx.folds):
        fit_idx = fold["fit"].intersection(fit_scope)
        es_idx = fold["es"].intersection(unit_rows)
        if len(es_idx) < 20:
            es_idx = fold["es"].intersection(fit_scope)
        ev_idx = fold["eval"].intersection(unit_rows)
        if len(ev_idx) == 0 or len(fit_idx) == 0:
            continue
        w = None if weight_fn is None else weight_fn(fit_idx)
        model = fit_cb(ctx, params, features, ctx.X_dev.loc[fit_idx], ctx.y_dev.loc[fit_idx],
                       ctx.X_dev.loc[es_idx], ctx.y_dev.loc[es_idx], weight=w)
        iters.append(max(1, (model.get_best_iteration() or 0) + 1))
        pred = predict_cb(ctx, model, params, ctx.X_dev.loc[ev_idx], features)
        if pred.ndim == 2:
            oof_q.loc[ev_idx, :] = pred
            pred = pred[:, list(quantile_alphas).index(0.5)]
        oof.loc[ev_idx] = pred
        a = ctx.y_dev.loc[ev_idx].to_numpy()
        maes.append(mean_absolute_error(a, pred))
        wapes.append(feeder_wape(a, pred, ctx.meta_dev.loc[ev_idx, "feeder"].to_numpy()))
        if importance:
            imps.append(pd.Series(model.get_feature_importance(
                data=make_pool(ctx, ctx.X_dev.loc[ev_idx], a, features), type="LossFunctionChange"),
                index=features))
        if fold_callback is not None:
            fold_callback(k, oof)
    done = oof.dropna()
    return {
        "unit": unit.name, "oof": oof, "oof_quantiles": oof_q,
        "mae": float(np.mean(maes)),
        "mae_se": float(np.std(maes, ddof=1) / np.sqrt(len(maes))) if len(maes) > 1 else 0.0,
        "feeder_wape": float(np.mean(wapes)),
        "best_iteration": int(np.median(iters)),
        "importance": pd.concat(imps, axis=1).mean(axis=1) if imps else None,
        "n_features": len(features),
        "sum_ratio": float(done.sum() / ctx.y_dev.loc[done.index].sum()),
        "under_rate": float((done < ctx.y_dev.loc[done.index]).mean()),
    }


def combine_predictions(pred_by_unit: dict, index) -> pd.Series:
    out = pd.concat(list(pred_by_unit.values())).reindex(index)
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} rows have no prediction.")
    return out


def iterations_for_refit(ctx: DevContext, cv_best_iteration, fit_splits) -> int:
    refit_rows = sum(len(ctx.full_x[s]) for s in fit_splits)
    fold_rows = len(ctx.X_dev) * (1 - 1 / ctx.cfg["n_folds"]) * (1 - ctx.cfg["es_feeder_share"])
    return int(max(50, round(cv_best_iteration * min(1.3, (refit_rows / fold_rows) ** 0.5))))


def refit_and_predict(ctx: DevContext, spec: dict, units: List[ModelUnit], fit_splits=("train", "valid"),
                      predict_split="test"):
    """spec[unit.name] = dict(features, params, iterations, [weighting=(TierWeighting, strength)], [quantile_alphas])."""
    X_all = pd.concat([ctx.full_x[s] for s in fit_splits])
    y_all = pd.concat([ctx.full_y[s] for s in fit_splits]).reindex(X_all.index)
    m_all = pd.concat([ctx.full_meta[s] for s in fit_splits]).reindex(X_all.index)
    X_p, m_p = ctx.full_x[predict_split], ctx.full_meta[predict_split]
    preds, models, qpreds = {}, {}, {}
    for u in units:
        s = spec[u.name]
        fm = scope_mask(m_all, u, ctx.area_class_col, u.train_scope)
        pidx = X_p.index[unit_mask(m_p, u, ctx.area_class_col)]
        if len(pidx) == 0:
            continue
        X_f, y_f, m_f = X_all[fm], y_all[fm], m_all[fm]
        w = None
        if s.get("weighting"):
            tw, strength = s["weighting"]
            w = tw.row_weights(X_f, y_f, m_f, strength)
        model = fit_cb(ctx, s["params"], s["features"], X_f, y_f, iterations=s["iterations"], weight=w)
        raw = predict_cb(ctx, model, s["params"], X_p.loc[pidx], s["features"])
        if raw.ndim == 2:
            alphas = list(s["quantile_alphas"])
            qpreds[u.name] = pd.DataFrame(raw, index=pidx, columns=alphas)
            raw = raw[:, alphas.index(0.5)]
        preds[u.name], models[u.name] = pd.Series(raw, index=pidx), model
    q_out = pd.concat(qpreds.values()).reindex(X_p.index) if qpreds else None
    return combine_predictions(preds, X_p.index), models, q_out


# =============================================================================
# 9. METRICS
# =============================================================================
def quadratic_weighted_kappa(t_true, t_pred, n_tiers) -> float:
    t_true, t_pred = np.asarray(t_true, int), np.asarray(t_pred, int)
    cm = np.bincount(t_true * n_tiers + t_pred, minlength=n_tiers ** 2).reshape(n_tiers, n_tiers).astype(float)
    if cm.sum() == 0:
        return np.nan
    w = np.subtract.outer(np.arange(n_tiers), np.arange(n_tiers)) ** 2 / (n_tiers - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    den = (w * expected).sum()
    return float(1 - (w * cm).sum() / den) if den > 0 else np.nan


def feeder_table(actual, pred, feeders) -> pd.DataFrame:
    fr = (pd.DataFrame({"feeder": feeders, "actual": actual, "predicted": pred})
          .groupby("feeder")
          .agg(span_count=("actual", "size"), actual_total_hours=("actual", "sum"),
               predicted_total_hours=("predicted", "sum")))
    fr["error_hours"] = fr["predicted_total_hours"] - fr["actual_total_hours"]
    fr["abs_pct_error"] = fr["error_hours"].abs() / fr["actual_total_hours"].replace(0, np.nan)
    return fr


def evaluate_combined(name, actual, pred, meta, hour_cuts) -> dict:
    """Span metrics per accessibility regime + combined; feeder metrics; HOUR-tier accuracy per regime."""
    actual = pd.Series(np.asarray(actual, float), index=meta.index)
    pred = pd.Series(np.asarray(pred, float), index=meta.index)
    row = {"Model": name, "Spans": len(actual)}
    for seg, flag in SEGMENTS.items():
        m = meta["inaccessible"].eq(flag).to_numpy()
        a, p = actual[m].to_numpy(), pred[m].to_numpy()
        ta, tp = tier_index(a, hour_cuts[seg]), tier_index(p, hour_cuts[seg])
        row.update({f"{seg}_MAE": mean_absolute_error(a, p), f"{seg}_R2": r2_score(a, p),
                    f"{seg}_Bias": float((p - a).mean()), f"{seg}_TierAcc": float((ta == tp).mean()),
                    f"{seg}_TierQWK": quadratic_weighted_kappa(ta, tp, len(hour_cuts[seg]) + 1)})
    row.update({"Combined_MAE": mean_absolute_error(actual, pred), "Combined_R2": r2_score(actual, pred),
                "Combined_Bias": float((pred - actual).mean())})
    ft = feeder_table(actual.to_numpy(), pred.to_numpy(), meta["feeder"].to_numpy())
    row.update({"Feeders": len(ft),
                "Feeder_R2": r2_score(ft["actual_total_hours"], ft["predicted_total_hours"]),
                "Feeder_MAE_Hours": ft["error_hours"].abs().mean(),
                "Feeder_WAPE": ft["error_hours"].abs().sum() / ft["actual_total_hours"].sum(),
                "Feeder_MdAPE": ft["abs_pct_error"].median(),
                "Feeder_Total_Bias_Pct": ft["error_hours"].sum() / ft["actual_total_hours"].sum()})
    for band in (0.15, 0.20, 0.25):
        row[f"Feeder_Within_{int(band * 100)}pct"] = ft["abs_pct_error"].le(band).mean()
    return row


COMPARE_COLS = ["Accessible_MAE", "Inaccessible_MAE", "Combined_MAE", "Combined_R2", "Combined_Bias",
                "Feeder_R2", "Feeder_WAPE", "Feeder_MdAPE", "Feeder_Within_15pct", "Feeder_Within_20pct",
                "Feeder_Total_Bias_Pct", "Accessible_TierAcc", "Accessible_TierQWK",
                "Inaccessible_TierAcc", "Inaccessible_TierQWK"]


def evaluate_units(actual, pred, meta, units, area_class_col) -> pd.DataFrame:
    rows = []
    for u in units:
        m = unit_mask(meta, u, area_class_col)
        if not m.any():
            continue
        a, p = np.asarray(actual, float)[m], np.asarray(pred, float)[m]
        rows.append({"Unit": u.name, "Spans": int(m.sum()), "MAE": mean_absolute_error(a, p),
                     "Bias": float((p - a).mean()), "Sum_Ratio": float(p.sum() / a.sum()),
                     "Under_Rate": float((p < a).mean())})
    return pd.DataFrame(rows).set_index("Unit")


def selection_score(row, ref_row, cfg) -> float:
    return (cfg["w_span_mae"] * row["Combined_MAE"] / ref_row["Combined_MAE"]
            + cfg["w_feeder_wape"] * row["Feeder_WAPE"] / ref_row["Feeder_WAPE"])


def scale_by_unit(pred: pd.Series, meta: pd.DataFrame, factors: dict, units, area_class_col) -> pd.Series:
    out = pred.copy()
    for u in units:
        m = unit_mask(meta, u, area_class_col)
        out[m] = out[m] * factors.get(u.name, 1.0)
    return out


# =============================================================================
# 10. TIER SCHEMES (hours <-> volume) + SAMPLE WEIGHTS
# =============================================================================
@dataclass
class TierScheme:
    """4 tiers per accessibility regime.

    hours : tier from HOURS. At bid time the span is priced in the tier of its *prediction*.
    volume: tier from the pre-work LiDAR intrusion volume -> known at bid time, no prediction needed.
    """
    name: str
    cuts: Dict[str, List[float]]
    labels: List[str] = field(default_factory=lambda: list(TIER_LABELS))
    volume_col: str = VOLUME_COL

    def _by_segment(self, values, meta):
        out = np.zeros(len(meta), int)
        flags = meta["inaccessible"].to_numpy()
        values = np.asarray(values, float)
        for seg, flag in SEGMENTS.items():
            m = flags == flag
            out[m] = tier_index(values[m], self.cuts[seg])
        return out

    def true_tier(self, X, y, meta) -> np.ndarray:
        src = y if self.name == "hours" else X[self.volume_col]
        return self._by_segment(src, meta)

    def bid_tier(self, X, pred, meta) -> np.ndarray:
        src = pred if self.name == "hours" else X[self.volume_col]
        return self._by_segment(src, meta)

    def bid_labels(self, X, pred, meta) -> np.ndarray:
        return np.array(self.labels)[self.bid_tier(X, pred, meta)]

    def to_dict(self):
        return {"name": self.name, "cuts": {k: [float(c) for c in v] for k, v in self.cuts.items()},
                "labels": self.labels, "volume_col": self.volume_col}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], {k: list(v) for k, v in d["cuts"].items()}, list(d["labels"]), d["volume_col"])


def volume_quartile_cuts(X_train: pd.DataFrame, meta_train: pd.DataFrame, scope="global",
                         col=VOLUME_COL) -> Dict[str, List[float]]:
    """Quarter bins of intrusion volume, fit on TRAIN only. scope='global' (same cuts for both regimes)
    or 'segment' (separate quartiles per accessibility regime)."""
    qs = [0.25, 0.50, 0.75]
    if scope == "global":
        cuts = [float(v) for v in np.quantile(X_train[col].astype(float), qs)]
        return {seg: cuts for seg in SEGMENTS}
    return {seg: [float(v) for v in np.quantile(
        X_train.loc[meta_train["inaccessible"].eq(flag).to_numpy(), col].astype(float), qs)]
        for seg, flag in SEGMENTS.items()}


@dataclass
class TierWeighting:
    """Per (regime, tier) weights = 1 + strength * badness, badness in [0, 1] from OOF performance.

    hours : badness = 1 - recall of the true hour tier (tiers the model fails to place).
    volume: badness = relative MAE + relative under-prediction of the volume tier.
    Weights are tier-level scalars derived from OOF predictions (like the tier cuts); they are
    applied to FIT rows only - early-stopping and scoring rows are unweighted.
    """
    scheme: TierScheme
    badness: Dict[str, List[float]]

    @classmethod
    def from_oof(cls, scheme: TierScheme, X, y, pred, meta):
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        tt = scheme.true_tier(X, y, meta)
        tp = scheme.bid_tier(X, pred, meta)
        flags = meta["inaccessible"].to_numpy()
        badness = {}
        for seg, flag in SEGMENTS.items():
            vals = []
            for k in range(len(scheme.labels)):
                m = (flags == flag) & (tt == k)
                if not m.any():
                    vals.append(0.0)
                elif scheme.name == "hours":
                    vals.append(float(1 - (tp[m] == k).mean()))
                else:
                    mean_k = max(y[m].mean(), 1e-6)
                    vals.append(float(np.abs(pred[m] - y[m]).mean() / mean_k
                                      + max(0.0, -(pred[m] - y[m]).mean()) / mean_k))
            vals = np.asarray(vals)
            badness[seg] = (vals / vals.max() if vals.max() > 0 else vals).round(4).tolist()
        return cls(scheme, badness)

    def row_weights(self, X, y, meta, strength) -> np.ndarray:
        if not strength:
            return np.ones(len(X))
        tiers = self.scheme.true_tier(X, y, meta)
        flags = meta["inaccessible"].to_numpy()
        w = np.ones(len(X))
        for seg, flag in SEGMENTS.items():
            m = flags == flag
            w[m] = 1.0 + strength * np.asarray(self.badness[seg])[tiers[m]]
        return w / w.mean()          # keep the average weight at 1

    def weight_table(self, strength) -> pd.DataFrame:
        return pd.DataFrame({seg: 1.0 + strength * np.asarray(b) for seg, b in self.badness.items()},
                            index=self.scheme.labels)

    def to_dict(self):
        return {"scheme": self.scheme.to_dict(), "badness": self.badness}

    @classmethod
    def from_dict(cls, d):
        return cls(TierScheme.from_dict(d["scheme"]), {k: list(v) for k, v in d["badness"].items()})


def class_wape(actual, pred, class_keys) -> float:
    """Sum over bid-time classes of |sum pred - sum actual| / total actual (pricing-table error)."""
    fr = pd.DataFrame({"a": actual, "p": pred, "k": class_keys}).groupby("k")[["a", "p"]].sum()
    return float((fr["p"] - fr["a"]).abs().sum() / max(fr["a"].sum(), 1e-9))


def class_keys(meta, area_class_col, bid_tiers) -> np.ndarray:
    return (meta["inaccessible"].astype(str).to_numpy() + "|" + meta[area_class_col].astype(str).to_numpy()
            + "|" + np.asarray(bid_tiers).astype(str))

import dataclasses

def with_conditional_quantiles(units, name_prefix: str, alphas: Sequence[float]):
    """
    Return a new list of Unit specs with `conditional_quantiles` overridden
    for every unit whose name starts with `name_prefix`. Handles frozen
    dataclasses, namedtuples, and plain mutable objects defensively.
    """
    out = []
    for u in units:
        if u.name.startswith(name_prefix):
            if dataclasses.is_dataclass(u):
                u = dataclasses.replace(u, conditional_quantiles=tuple(alphas))
            elif hasattr(u, "_replace"):
                u = u._replace(conditional_quantiles=tuple(alphas))
            else:
                u.conditional_quantiles = tuple(alphas)
        out.append(u)
    return out


def fit_tier_conditional_bias(oof_pred: pd.Series, y: pd.Series, meta: pd.DataFrame,
                               scheme, X: pd.DataFrame, min_rows: int = 60,
                               shrinkage_rows: int = 150, max_abs_adjustment: float = 2.0) -> pd.DataFrame:
    """
    Leak-safe additive correction learned from OOF residuals, grouped by
    (accessibility, PREDICTED tier). Targets systematic under-tiering at
    the top of the distribution - e.g. Inaccessible spans that are truly
    Heavy but consistently predicted into Medium-Heavy. Fit on OOF only;
    grouping uses the model's own predicted tier, so it directly targets
    the boundary spans causing the tier confusion.
    """
    pred_tier = np.asarray(scheme.bid_labels(X, oof_pred, meta))
    frame = pd.DataFrame({
        "inaccessible": meta["inaccessible"].reindex(oof_pred.index).to_numpy(),
        "pred_tier": pred_tier,
        "actual": y.reindex(oof_pred.index).to_numpy(),
        "pred": oof_pred.to_numpy(),
    }, index=oof_pred.index)
    frame["residual"] = frame["actual"] - frame["pred"]

    rows = []
    for (flag, tier), g in frame.groupby(["inaccessible", "pred_tier"], observed=True):
        n = len(g)
        raw = float(g["residual"].median())
        weight = n / (n + shrinkage_rows)
        adj = float(np.clip(raw * weight, -max_abs_adjustment, max_abs_adjustment)) if n >= min_rows else 0.0
        rows.append({"inaccessible": int(flag), "pred_tier": str(tier), "N": n,
                     "Raw_Median_Residual": raw, "Shrinkage_Weight": weight,
                     "Adjustment": adj, "Low_Sample_Flag": n < min_rows})
    return pd.DataFrame(rows)


def apply_tier_conditional_bias(pred: pd.Series, meta: pd.DataFrame, calibration: pd.DataFrame,
                                 scheme, X: pd.DataFrame) -> pd.Series:
    """Apply a fit_tier_conditional_bias() table to new predictions, using the
    SAME scheme + the current `pred` to assign predicted tiers - consistent
    whether called on Test or Holdout."""
    pred_tier = np.asarray(scheme.bid_labels(X, pred, meta))
    lookup = calibration.set_index(["inaccessible", "pred_tier"])["Adjustment"]
    keys = pd.MultiIndex.from_arrays([
        pd.to_numeric(meta["inaccessible"], errors="coerce").astype("Int64"),
        pd.Series(pred_tier, index=meta.index).astype(str),
    ])
    delta = lookup.reindex(keys).fillna(0.0).to_numpy(float)
    return (pred + delta).clip(lower=0)


def heavy_tier_recall_table(y: pd.Series, pred: pd.Series, meta: pd.DataFrame, scheme,
                             X: pd.DataFrame, tier_labels: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Recall/precision for the TOP tier specifically, by accessibility segment.
    Surfaces top-tier under-prediction even when aggregate metrics
    (sum_ratio, under_rate, Combined_MAE) look acceptable - which is
    exactly what happened here: Weighted(hours) looked like a modest
    aggregate improvement while Inaccessible-Heavy recall stayed <10%.
    """
    tier_labels = list(tier_labels or TIER_LABELS)
    top = tier_labels[-1]
    one_below = tier_labels[-2] if len(tier_labels) > 1 else top

    true_lab = pd.Series(np.asarray(scheme.true_tier(X, y, meta)), index=meta.index).map(
        dict(enumerate(tier_labels)))
    pred_lab = pd.Series(np.asarray(scheme.bid_labels(X, pred, meta)), index=meta.index)

    rows = []
    for seg, flag in SEGMENTS.items():
        m = meta["inaccessible"].eq(flag).to_numpy()
        tt, pt = true_lab[m], pred_lab[m]
        is_top = tt.eq(top)
        n_top = int(is_top.sum())
        recall = float((is_top & pt.eq(top)).sum() / n_top) if n_top else np.nan
        precision = float((is_top & pt.eq(top)).sum() / max(int(pt.eq(top).sum()), 1))
        one_low_share = float((is_top & pt.eq(one_below)).sum() / n_top) if n_top else np.nan
        rows.append({"Segment": seg, f"{top}_N": n_top, f"{top}_Recall": recall,
                     f"{top}_Precision": precision, "Predicted_One_Tier_Low_Share": one_low_share})
    return pd.DataFrame(rows).set_index("Segment")

# =============================================================================
# 11. OBJECTIVES + OPTUNA
# =============================================================================
DEFAULT_TIER_LOSS_CFG = {"under_weight": 1.5, "distance_power": 2.0, "tau": 0.25, "mae_weight": 0.25}
DEFAULT_SCHEME_OBJECTIVE_WEIGHTS = {
    "hours": {"mae": 0.35, "feeder_wape": 0.20, "tier_loss": 0.25, "class_wape": 0.20},
    "volume": {"mae": 0.45, "feeder_wape": 0.25, "tier_loss": 0.0, "class_wape": 0.30},
}


def tier_loss_v2(y_true, y_pred, cuts, *, under_weight=1.5, distance_power=2.0, tau=0.25,
                 mae_weight=0.25, soft=True) -> float:
    """Ordinal-distance, under-tiering-weighted, smooth hour-tier loss anchored with relative MAE (lower=better)."""
    y_true, y_pred, cuts = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(cuts, float)
    n = len(cuts) + 1
    tt = tier_index(y_true, cuts)
    tp = expit((y_pred[:, None] - cuts[None, :]) / tau).sum(1) if soft else tier_index(y_pred, cuts).astype(float)
    gap = tp - tt
    w = np.where(gap < 0, under_weight, 1.0)
    ordinal = np.mean(w * np.abs(gap) ** distance_power) / (n - 1) ** distance_power
    naive = np.mean(np.abs(y_true - np.median(y_true))) or 1.0
    return float((1 - mae_weight) * ordinal + mae_weight * np.mean(np.abs(y_true - y_pred)) / naive)


def raw_objective_terms(ctx: DevContext, idx, pred, scheme: Optional[TierScheme]) -> dict:
    y = ctx.y_dev.loc[idx].to_numpy()
    meta = ctx.meta_dev.loc[idx]
    terms = {"mae": float(np.mean(np.abs(y - pred))),
             "feeder_wape": feeder_wape(y, pred, meta["feeder"].to_numpy())}
    if scheme is not None:
        X = ctx.X_dev.loc[idx]
        terms["class_wape"] = class_wape(y, pred, class_keys(meta, ctx.area_class_col, scheme.bid_tier(X, pred, meta)))
        tl = 0.0
        if scheme.name == "hours":
            flags = meta["inaccessible"].to_numpy()
            for seg, flag in SEGMENTS.items():
                m = flags == flag
                if m.any():
                    tl += m.mean() * tier_loss_v2(y[m], pred[m], scheme.cuts[seg], **DEFAULT_TIER_LOSS_CFG)
        terms["tier_loss"] = tl
    return terms


def make_objective_fn(ctx: DevContext, ref_terms: dict, scheme: Optional[TierScheme] = None, weights=None):
    """objective(idx, pred) -> float, each term divided by its reference value (lower = better).
    scheme=None -> the plain span-MAE + feeder-WAPE composite."""
    if scheme is None:
        weights = weights or {"mae": ctx.cfg["w_span_mae"], "feeder_wape": ctx.cfg["w_feeder_wape"]}
    else:
        weights = weights or DEFAULT_SCHEME_OBJECTIVE_WEIGHTS[scheme.name]

    def objective(idx, pred):
        terms = raw_objective_terms(ctx, idx, np.asarray(pred, float), scheme)
        return float(sum(w * terms[k] / max(ref_terms.get(k, 1.0), 1e-9) for k, w in weights.items() if w))
    return objective


def suggest_tree_params(trial, max_iterations):
    gp = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])
    p = {
        "iterations": max_iterations,
        "depth": trial.suggest_int("depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 4, 10]),
        "grow_policy": gp,
    }
    if gp == "Depthwise":
        p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100, log=True)
    return p


def params_from_trial(trial_params, families, max_iterations, loss_override=None):
    fixed = optuna.trial.FixedTrial(trial_params)
    p = suggest_tree_params(fixed, max_iterations)
    p["loss_function"] = loss_override or make_loss_builder(families)(fixed)
    return p


def make_study_objective(ctx: DevContext, unit: ModelUnit, features, families, objective_fn,
                         weighting: Optional[TierWeighting] = None):
    builder = make_loss_builder(families)

    def objective(trial):
        params = suggest_tree_params(trial, ctx.cfg["max_iterations"])
        params["loss_function"] = builder(trial)
        weight_fn = None
        if weighting is not None:
            strength = trial.suggest_float("weight_strength", 0.0, 3.0)
            if strength > 0:
                weight_fn = lambda idx: weighting.row_weights(ctx.X_dev.loc[idx], ctx.y_dev.loc[idx],  # noqa: E731
                                                              ctx.meta_dev.loc[idx], strength)

        def report(k, oof):
            d = oof.dropna()
            trial.report(objective_fn(d.index, d.to_numpy()), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        r = cv_unit(ctx, unit, features, params, weight_fn=weight_fn, fold_callback=report)
        trial.set_user_attr("best_iteration", r["best_iteration"])
        trial.set_user_attr("cv_mae", r["mae"])
        trial.set_user_attr("sum_ratio", r["sum_ratio"])
        return objective_fn(r["oof"].index, r["oof"].to_numpy())
    return objective


def run_study(name, objective, n_trials, seed_params, random_state=RANDOM_STATE, show_progress=True):
    study = optuna.create_study(
        direction="minimize", study_name=name,
        sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=min(10, n_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(8, n_trials), n_warmup_steps=1),
    )
    study.enqueue_trial(seed_params, skip_if_exists=True)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=show_progress)
    return study


def default_seed_params(ref_params: dict, loss_family: str) -> dict:
    return {"grow_policy": "SymmetricTree", "depth": int(np.clip(ref_params.get("depth", 6), 4, 9)),
            "learning_rate": float(np.clip(ref_params.get("learning_rate", 0.05), 0.015, 0.15)),
            "l2_leaf_reg": float(np.clip(ref_params.get("l2_leaf_reg", 5.0), 1.0, 30.0)),
            "loss_family": loss_family}


def underprediction_flag(cv_result: dict, cfg: dict) -> bool:
    return (cv_result["sum_ratio"] < 1 - cfg["underpred_ratio_tol"]
            or cv_result["under_rate"] > cfg["underpred_rate_tol"])


def spec_is_current(path, expected: dict) -> bool:
    """expected = {unit_name: {"features": [...], ...}} - every listed key must match the saved file."""
    if not os.path.exists(path):
        return False
    saved = read_json(path)
    return all(all(saved.get(u, {}).get(k) == v for k, v in keys.items()) for u, keys in expected.items())


# =============================================================================
# 12. AREA COLUMN SELECTION + FEATURE PRUNING
# =============================================================================
def select_area_column(ctx: DevContext, ref_params: dict, hour_cuts: dict):
    """Each candidate area column is added alone; accessible/inaccessible pair; OOF combined."""
    units = segment_units()
    base = [c for c in ctx.X_dev.columns if c not in AREA_CANDIDATE_COLS]
    cands = {"none": base, **{c: base + [c] for c in AREA_CANDIDATE_COLS if c in ctx.X_dev.columns}}
    rows, test_rows = [], []
    for name, feats in cands.items():
        res = {u.name: cv_unit(ctx, u, feats, ref_params) for u in units}
        oof = combine_predictions({k: r["oof"] for k, r in res.items()}, ctx.X_dev.index)
        rows.append(evaluate_combined(name, ctx.y_dev, oof, ctx.meta_dev, hour_cuts))
        if ctx.cfg["report_test_for_experiments"] and name != "none":
            spec = {u.name: {"features": feats, "params": ref_params,
                             "iterations": iterations_for_refit(ctx, res[u.name]["best_iteration"], ("train", "valid"))}
                    for u in units}
            pred, _, _ = refit_and_predict(ctx, spec, units)
            test_rows.append(evaluate_combined(name, ctx.full_y["test"], pred, ctx.full_meta["test"], hour_cuts))
        print(f"  {name:28s} CV MAE={rows[-1]['Combined_MAE']:.4f} feeder R2={rows[-1]['Feeder_R2']:.4f} "
              f"feeder WAPE={rows[-1]['Feeder_WAPE']:.4f}")
    table = pd.DataFrame(rows).set_index("Model")
    table["Selection_Score"] = table.apply(lambda r: selection_score(r, table.loc["none"], ctx.cfg), axis=1)
    candidates = [c for c in AREA_CANDIDATE_COLS if c in table.index]
    chosen = table.loc[candidates, "Selection_Score"].idxmin()
    test_table = pd.DataFrame(test_rows).set_index("Model") if test_rows else None
    return chosen, table, test_table

def resolve_area_column(cv_table: pd.DataFrame, override: Optional[str] = None,
                         candidate_cols: Optional[List[str]] = None):
    """
    Resolve which area column to use for pricing/segmentation.

    Returns (chosen_col, auto_chosen_col, override_used: bool).
    `auto_chosen_col` is always the CV Selection_Score argmin, so the
    audit trail shows what auto-selection would have picked even when
    an override is used.
    """
    candidate_cols = candidate_cols or AREA_CANDIDATE_COLS
    valid = [c for c in candidate_cols if c in cv_table.index]
    auto_chosen = cv_table.loc[valid, "Selection_Score"].idxmin()

    if override is None:
        return auto_chosen, auto_chosen, False

    if override not in candidate_cols:
        raise ValueError(
            f"AREA_COLUMN_OVERRIDE={override!r} is not one of "
            f"AREA_CANDIDATE_COLS={candidate_cols}."
        )
    if override not in cv_table.index:
        raise ValueError(
            f"AREA_COLUMN_OVERRIDE={override!r} was not scored in the CV "
            f"area table (scored: {list(cv_table.index)})."
        )
    return override, auto_chosen, True


def prune_unit_features(ctx: DevContext, unit: ModelUnit, features, ref_params, protected, groups,
                        corr_threshold=0.98, shares=(0.20, 0.35, 0.50)):
    """Group ablations + OOF LossFunctionChange drops; 1-SE rule picks the smallest good set."""
    unit_X = ctx.X_dev.loc[ctx.rows(unit, unit.train_scope)]

    def drop(feats, to_drop):
        to_drop = set(to_drop) - set(protected)
        return [c for c in feats if c not in to_drop]

    constants = [c for c in features if unit_X[c].nunique(dropna=False) <= 1]
    all_f = drop(features, constants)
    base = cv_unit(ctx, unit, all_f, ref_params, importance=True)
    imp = base["importance"].sort_values()
    ranked = [c for c in imp.index if c not in protected]

    num = [c for c in all_f if c not in ctx.categorical_pool]
    corr = unit_X[num].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    corr_drop = set()
    for a, b in zip(*np.where(upper.to_numpy() > corr_threshold)):
        fa, fb = upper.index[a], upper.columns[b]
        corr_drop.add(fa if imp.get(fa, 0) < imp.get(fb, 0) else fb)

    exps = {
        "all": all_f,
        "-alley": drop(all_f, groups["alley"]),
        "-road_core": drop(all_f, groups["road_core"]),
        "-road_all(incl alley)": drop(all_f, groups["road_core"] + groups["alley"]),
        "-land_cover": drop(all_f, groups["land_cover_numeric"] + groups["land_cover_categorical"]),
        "-corr_duplicates": drop(all_f, corr_drop),
        "-nonpositive_importance": drop(all_f, imp[imp <= 0].index),
    }
    for sh in shares:
        exps[f"-bottom_{int(sh * 100)}pct"] = drop(all_f, ranked[: int(len(ranked) * sh)])

    rows, cvs = [], {"all": base}
    for name, feats in exps.items():
        r = base if name == "all" else cv_unit(ctx, unit, feats, ref_params)
        cvs[name] = r
        rows.append({"Experiment": name, "n_features": len(feats), "CV_MAE": r["mae"], "CV_MAE_SE": r["mae_se"],
                     "CV_Feeder_WAPE": r["feeder_wape"], "best_iteration": r["best_iteration"]})
    table = pd.DataFrame(rows).set_index("Experiment")
    table["Delta_MAE_vs_all"] = table["CV_MAE"] - table.loc["all", "CV_MAE"]
    best = table["CV_MAE"].idxmin()
    ok = table[table["CV_MAE"] <= table.loc[best, "CV_MAE"] + table.loc[best, "CV_MAE_SE"]]
    chosen = ok.sort_values(["n_features", "CV_MAE"]).index[0]
    table["Chosen"] = table.index == chosen
    missing = [c for c in protected if c not in exps[chosen]]
    assert not missing, f"Protected features dropped: {missing}"
    return {"selected": exps[chosen], "chosen": chosen, "table": table, "importance": imp,
            "cv": cvs, "constants": constants}


def plot_importance(importance_by_unit: dict, save_path=None, show_plot=False):
    n = len(importance_by_unit)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 9))
    for ax, (name, imp) in zip(np.atleast_1d(axes), importance_by_unit.items()):
        shown = pd.concat([imp.head(10), imp.tail(15)])
        ax.barh(shown.index, shown.values, color=["tab:red" if v <= 0 else "tab:blue" for v in shown])
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_title(f"{name}: OOF LossFunctionChange\n(red = no/negative value)", fontsize=10)
    finish_figure(fig, save_path, show_plot)


# =============================================================================
# 13. HOUR-TIER BOUNDARY SEARCH + PRICING MATRIX
# =============================================================================
DEFAULT_BOUNDARY_CFG = {"min_tier_share": 0.10, "min_gap": 1.0, "max_candidates": 40, "w_qwk": 0.6,
                        "w_eta": 0.4, "lower_q": 0.05, "upper_q": 0.95}


def candidate_cuts(y, cfg=DEFAULT_BOUNDARY_CFG):
    y = np.asarray(y, float)
    vals = np.sort(np.unique(y))
    mids = (vals[:-1] + vals[1:]) / 2          # midpoints between recorded (~0.5 h) values
    lo, hi = np.quantile(y, cfg["lower_q"]), np.quantile(y, cfg["upper_q"])
    mids = mids[(mids >= lo) & (mids <= hi)]
    if len(mids) > cfg["max_candidates"]:
        targets = np.quantile(y, np.linspace(cfg["lower_q"], cfg["upper_q"], cfg["max_candidates"]))
        mids = np.unique([mids[np.abs(mids - t).argmin()] for t in targets])
    return mids


def score_boundaries(y, pred, cuts) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(cuts) + 1
    ta, tp = tier_index(y, cuts), tier_index(pred, cuts)
    shares = np.bincount(ta, minlength=n) / len(y)
    means = np.array([y[ta == k].mean() if (ta == k).any() else np.nan for k in range(n)])
    ssb = sum((ta == k).sum() * (means[k] - y.mean()) ** 2 for k in range(n) if (ta == k).any())
    return {"cuts": tuple(float(c) for c in cuts), "tier_acc": float((ta == tp).mean()),
            "qwk": quadratic_weighted_kappa(ta, tp, n), "within_1_tier": float((np.abs(ta - tp) <= 1).mean()),
            "eta2": float(ssb / ((y - y.mean()) ** 2).sum()), "min_share": float(shares.min()),
            "shares": np.round(shares, 3).tolist(), "tier_mean_hours": np.round(means, 2).tolist()}


def recommend_boundaries(y, pred, n_tiers=4, cfg=DEFAULT_BOUNDARY_CFG) -> pd.DataFrame:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rows = []
    for cuts in combinations(candidate_cuts(y, cfg), n_tiers - 1):
        if np.min(np.diff(cuts)) < cfg["min_gap"]:
            continue
        if (np.bincount(tier_index(y, cuts), minlength=n_tiers) / len(y)).min() < cfg["min_tier_share"]:
            continue
        rows.append(score_boundaries(y, pred, cuts))
    if not rows:
        raise ValueError("No boundary set satisfies the constraints; relax the boundary config.")
    t = pd.DataFrame(rows)
    t["rec_score"] = cfg["w_qwk"] * t["qwk"] + cfg["w_eta"] * t["eta2"]
    return t.sort_values("rec_score", ascending=False).reset_index(drop=True)


def plot_tier_boundaries(y_by_seg, pred_by_seg, cuts_now, cuts_prev, save_path=None, show_plot=False):
    fig, axes = plt.subplots(1, len(y_by_seg), figsize=(8 * len(y_by_seg), 5))
    for ax, seg in zip(np.atleast_1d(axes), y_by_seg):
        up = np.quantile(y_by_seg[seg], 0.99)
        ax.hist(np.clip(y_by_seg[seg], None, up), bins=60, alpha=0.5, label="actual")
        ax.hist(np.clip(pred_by_seg[seg], None, up), bins=60, alpha=0.5, label="predicted")
        for c in cuts_prev[seg]:
            ax.axvline(c, color="grey", linestyle=":", linewidth=1.5)
        for c in cuts_now[seg]:
            ax.axvline(c, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{seg}: grey = previous cuts, red = cuts in use")
        ax.legend()
    finish_figure(fig, save_path, show_plot)


def plot_volume_tiers(X, y, meta, scheme: TierScheme, save_path=None, show_plot=False):
    """Hours distribution inside each volume quartile (does volume separate workload?)."""
    tiers = scheme.true_tier(X, y, meta)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (seg, flag) in zip(axes, SEGMENTS.items()):
        m = meta["inaccessible"].eq(flag).to_numpy()
        data = [np.asarray(y, float)[m & (tiers == k)] for k in range(len(scheme.labels))]
        ax.boxplot(data, labels=scheme.labels, showfliers=False)
        ax.set_title(f"{seg}: hours by volume tier (cuts {[round(c, 1) for c in scheme.cuts[seg]]})")
        ax.set_ylabel("Manhours")
    finish_figure(fig, save_path, show_plot)


# =============================================================================
# 14. 24-CLASS RISK TABLE (accessibility x area x veg tier)
# =============================================================================
RISK_WEIGHTS = {"width": 0.30, "cv": 0.25, "mae": 0.25, "under_bias": 0.20}
INVESTIGATE_CFG = {"quantile": 0.75, "min_flags": 2, "risk_threshold": 1.25, "min_n": 30}


def bootstrap_mean_ci(values, n_boot=2000, level=0.95, seed=RANDOM_STATE):
    v = np.asarray(values, float)
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, int(2e6 // len(v)))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        means[s:e] = v[rng.integers(0, len(v), size=(e - s, len(v)))].mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2]))


def class_risk_table(actual, point, q_lo, q_mid, q_hi, meta, area_class_col, tier_labels_per_span,
                     tier_order=TIER_LABELS, n_boot=2000, risk_weights=RISK_WEIGHTS,
                     investigate_cfg=INVESTIGATE_CFG) -> pd.DataFrame:
    """24 bid-time classes: Accessibility (2) x Area (3) x veg tier (4).

    Hours statistics (N, mean, median, 95% CI of mean, CV) describe ACTUAL hours.
    P10 / P50 / P90 are the class means of the per-span PREDICTED quantiles; Pred_Width = mean(P90 - P10).
    MAE / Bias use the point prediction. Risk_Score: weighted ratio to the all-class value
    (1.0 = average; > 1 riskier). Under_Bias only counts under-prediction.
    """
    fr = pd.DataFrame({
        "Accessibility": np.where(meta["inaccessible"].to_numpy() == 1, "Inaccessible", "Accessible"),
        "Area": meta[area_class_col].astype(str).to_numpy(),
        "Veg_Tier": np.asarray(tier_labels_per_span).astype(str),
        "actual": np.asarray(actual, float), "point": np.asarray(point, float),
        "q_lo": np.asarray(q_lo, float), "q_mid": np.asarray(q_mid, float), "q_hi": np.asarray(q_hi, float),
    })
    fr["width"] = fr["q_hi"] - fr["q_lo"]
    fr["abs_err"] = (fr["point"] - fr["actual"]).abs()
    fr["in_band"] = (fr["actual"] >= fr["q_lo"]) & (fr["actual"] <= fr["q_hi"])
    unmapped = int((~fr["Area"].isin(AREA_CLASSES)).sum())

    rows = []
    for key, g in fr[fr["Area"].isin(AREA_CLASSES)].groupby(["Accessibility", "Area", "Veg_Tier"]):
        lo, hi = bootstrap_mean_ci(g["actual"], n_boot=n_boot)
        mean = g["actual"].mean()
        rows.append({
            "Accessibility": key[0], "Area": key[1], "Veg_Tier": key[2], "N": len(g),
            "Mean_Hours": mean, "Median_Hours": g["actual"].median(),
            "Mean_CI95_Low": lo, "Mean_CI95_High": hi,
            "P10": g["q_lo"].mean(), "P50": g["q_mid"].mean(), "P90": g["q_hi"].mean(),
            "Pred_Width": g["width"].mean(), "Pred_Mean": g["point"].mean(),
            "MAE": g["abs_err"].mean(), "Bias": (g["point"] - g["actual"]).mean(),
            "CV": g["actual"].std(ddof=1) / mean if mean > 0 and len(g) > 1 else np.nan,
            "Coverage_80": g["in_band"].mean(),
        })
    idx = pd.MultiIndex.from_product([list(SEGMENTS), AREA_CLASSES, list(tier_order)],
                                     names=["Accessibility", "Area", "Veg_Tier"])
    t = pd.DataFrame(rows).set_index(["Accessibility", "Area", "Veg_Tier"]).reindex(idx)
    t["N"] = t["N"].fillna(0).astype(int)

    ref_width = fr["width"].mean()
    ref_cv = fr["actual"].std(ddof=1) / fr["actual"].mean()
    ref_mae = fr["abs_err"].mean()
    under = (-t["Bias"]).clip(lower=0)
    t["Risk_Score"] = (risk_weights["width"] * t["Pred_Width"] / ref_width
                       + risk_weights["cv"] * t["CV"] / ref_cv
                       + risk_weights["mae"] * t["MAE"] / ref_mae
                       + risk_weights["under_bias"] * under / ref_mae)
    t["Low_N"] = t["N"] < investigate_cfg["min_n"]

    ok = t[~t["Low_N"]]
    flags = pd.DataFrame({c: t[c] >= ok[c].quantile(investigate_cfg["quantile"]) for c in ["Pred_Width", "CV", "MAE"]})
    t["High_Var_Flags"] = flags.sum(axis=1).astype(int)
    t["Flagged_On"] = flags.apply(lambda r: ",".join(c for c in flags.columns if r[c]), axis=1)
    t["Investigate"] = (~t["Low_N"]) & (t["N"] > 0) & (
        (t["High_Var_Flags"] >= investigate_cfg["min_flags"]) | (t["Risk_Score"] >= investigate_cfg["risk_threshold"]))
    t.attrs["unmapped_spans"] = unmapped
    return t


def class_eta2(actual, class_labels) -> float:
    """Share of hour variance explained by the classes (higher = more homogeneous class prices)."""
    fr = pd.DataFrame({"a": np.asarray(actual, float), "k": np.asarray(class_labels)})
    g = fr.groupby("k")["a"]
    ssb = (g.size() * (g.mean() - fr["a"].mean()) ** 2).sum()
    sst = ((fr["a"] - fr["a"].mean()) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def scheme_summary(tables: Dict[str, pd.DataFrame], actual, class_labels_by_scheme: dict) -> pd.DataFrame:
    rows = []
    for name, t in tables.items():
        v = t[t["N"] > 0]
        w = v["N"] / v["N"].sum()
        rows.append({
            "Scheme": name, "Eta2_24class": class_eta2(actual, class_labels_by_scheme[name]),
            # N-weighted span-level width / MAE are identical across schemes (same spans), so only
            # class-level quantities are compared here.
            "Weighted_CV": float((w * v["CV"].fillna(0)).sum()),
            "Weighted_Abs_Class_Bias": float((w * v["Bias"].abs()).sum()),
            "Weighted_Risk": float((w * v["Risk_Score"]).sum()),
            "Max_Risk": float(v.loc[~v["Low_N"], "Risk_Score"].max()),
            "Share_Spans_Flagged": float(v.loc[v["Investigate"], "N"].sum() / v["N"].sum()),
            "Classes_Investigate": int(t["Investigate"].sum()),
            "Classes_Low_N": int((t["Low_N"] & (t["N"] > 0)).sum()),
            "Empty_Classes": int((t["N"] == 0).sum()),
            "Min_N": int(v["N"].min()),
        })
    return pd.DataFrame(rows).set_index("Scheme")


RISK_TABLE_COLS = ["N", "Mean_Hours", "Median_Hours", "Mean_CI95_Low", "Mean_CI95_High", "P10", "P50", "P90",
                   "Pred_Width", "MAE", "Bias", "CV", "Coverage_80", "Risk_Score", "Investigate"]


def plot_risk_heatmap(table: pd.DataFrame, title, value="Risk_Score", save_path=None, show_plot=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    vmin, vmax = np.nanmin(table[value].to_numpy(float)), np.nanmax(table[value].to_numpy(float))  # shared scale
    for ax, seg in zip(axes, SEGMENTS):
        sub = table.loc[seg]
        grid = sub[value].unstack("Veg_Tier").reindex(index=AREA_CLASSES)
        grid = grid[[c for c in sub.index.get_level_values("Veg_Tier").unique()]]
        ann = grid.round(2).astype(str) + "\n" + sub["N"].unstack("Veg_Tier").reindex(
            index=AREA_CLASSES)[grid.columns].astype(int).map(lambda n: f"n={n}")
        inv = sub["Investigate"].unstack("Veg_Tier").reindex(index=AREA_CLASSES)[grid.columns].fillna(False)
        ann = ann.where(~inv.astype(bool), ann + " ⚑")
        sns.heatmap(grid.astype(float), annot=ann, fmt="", cmap="Reds", ax=ax, cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} | {seg} | {value} (⚑ = investigate)", fontsize=10)
    finish_figure(fig, save_path, show_plot)


# =============================================================================
# 15. OTHER PLOTS / TABLES
# =============================================================================
def pinball(y, q, alpha) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def multi_quantile_loss(alphas) -> str:
    return "MultiQuantile:alpha=" + ",".join(f"{a:g}" for a in alphas)


def quantile_calibration(y, qframe: pd.DataFrame, meta, units, area_class_col) -> pd.DataFrame:
    rows = []
    y = np.asarray(y, float)
    for u in units:
        m = unit_mask(meta, u, area_class_col)
        if not m.any():
            continue
        row = {"Unit": u.name, "N": int(m.sum())}
        for a in qframe.columns:
            q = qframe[a].to_numpy()[m]
            row[f"Cov(y<=q{int(a * 100)})"] = float((y[m] <= q).mean())
            row[f"Pinball_q{int(a * 100)}"] = pinball(y[m], q, a)
        lo, hi = qframe[min(qframe.columns)].to_numpy()[m], qframe[max(qframe.columns)].to_numpy()[m]
        row["Interval_Coverage"] = float(((y[m] >= lo) & (y[m] <= hi)).mean())
        row["Mean_Width"] = float((hi - lo).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Unit")


def plot_tier_confusion(ax, t_actual, t_pred, title, labels=TIER_LABELS, cmap="Blues"):
    n = len(labels)
    cm = np.bincount(np.asarray(t_actual) * n + np.asarray(t_pred), minlength=n * n).reshape(n, n)
    row = cm.sum(1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row != 0)
    annot = np.array([[f"{pct[i, j]:.0%}\n(n={cm[i, j]})" for j in range(n)] for i in range(n)])
    sns.heatmap(pct, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    ax.set_title(title, fontsize=10)


def confusion_grid(preds_by_model: dict, actual, meta, hour_cuts, split_label, save_path=None, show_plot=False):
    """Hour-tier confusion (actual hour tier vs predicted hour tier) per accessibility regime."""
    names = list(preds_by_model)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.6 * len(names)), squeeze=False)
    a_all = np.asarray(actual, float)
    for r, name in enumerate(names):
        p_all = np.asarray(preds_by_model[name], float)
        for c, (seg, flag) in enumerate(SEGMENTS.items()):
            m = meta["inaccessible"].eq(flag).to_numpy()
            ta, tp = tier_index(a_all[m], hour_cuts[seg]), tier_index(p_all[m], hour_cuts[seg])
            plot_tier_confusion(axes[r, c], ta, tp,
                                f"{split_label} | {name} | {seg}\nhour cuts {list(hour_cuts[seg])} | acc {np.mean(ta == tp):.1%}",
                                cmap="Blues" if seg == "Accessible" else "Purples")
    finish_figure(fig, save_path, show_plot)


def feeder_scatter(preds_by_model: dict, actual, meta, split_label, save_path=None, show_plot=False, band=0.15):
    names = list(preds_by_model)
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    tables = {n: feeder_table(np.asarray(actual, float), np.asarray(preds_by_model[n], float),
                              meta["feeder"].to_numpy()) for n in names}
    top = max(max(t["actual_total_hours"].max(), t["predicted_total_hours"].max()) for t in tables.values())
    for ax, n in zip(axes.ravel(), names):
        t = tables[n]
        r2 = r2_score(t["actual_total_hours"], t["predicted_total_hours"])
        ax.scatter(t["actual_total_hours"], t["predicted_total_hours"], alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.plot([0, top], [0, top], "r--", linewidth=1)
        ax.plot([0, top], [0, top * (1 + band)], ":", color="grey", linewidth=0.7)
        ax.plot([0, top], [0, top * (1 - band)], ":", color="grey", linewidth=0.7)
        ax.set_title(f"{split_label} | {n}\nfeeder R² = {r2:.3f} | within ±{band:.0%}: "
                     f"{t['abs_pct_error'].le(band).mean():.0%}", fontsize=10)
        ax.set_xlabel("Actual feeder manhours")
        ax.set_ylabel("Predicted feeder manhours")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    finish_figure(fig, save_path, show_plot)
    return tables


def issue_table(actual, pred, issues) -> pd.DataFrame:
    fr = pd.DataFrame({"issue": np.asarray(issues, object), "actual": np.asarray(actual, float),
                       "predicted": np.asarray(pred, float)})
    fr["abs_error"] = (fr["predicted"] - fr["actual"]).abs()
    return (fr.groupby("issue", dropna=False)
            .agg(n=("actual", "size"), MAE=("abs_error", "mean"),
                 Actual_Mean=("actual", "mean"), Predicted_Mean=("predicted", "mean"))
            .sort_values("MAE", ascending=False))

def confusion_grid_by_area(preds_by_model: dict, actual, meta, hour_cuts, area_col,
                            split_label, area_classes=None, save_path_template=None,
                            show_plot=False, min_n=1):
    """
    Hour-tier confusion (actual vs predicted), faceted by Accessibility x Area class.

    One figure PER MODEL is produced, with rows = area classes and columns =
    accessibility segments - e.g. 3 areas x 2 segments = 6 confusion panels
    per model, matching the 24-class (Accessibility x Area x Tier) framework
    already used elsewhere (risk_tables, class_risk_table).

    area_col: the column in `meta` holding the area class label
              (e.g. AREA_CLASS_COL, same variable used in risk_tables()).
    area_classes: ordered list of area labels to render as rows; defaults to
                  AREA_CLASSES (["Urban", "Suburban", "Rural"]) for
                  consistency with the rest of the pipeline's reporting.
    save_path_template: a path string containing a literal "{model}"
                  placeholder, e.g. fig("nn_holdout_confusion_by_area_{model}.png").
                  One file is written per model, with {model} substituted.
    min_n: area x segment cells with fewer than this many spans are rendered
           as an empty, labeled panel instead of a (potentially misleading)
           confusion matrix built from too few points.
    """
    names = list(preds_by_model)
    areas = area_classes or AREA_CLASSES
    a_all = np.asarray(actual, float)
    figs = {}

    for name in names:
        p_all = np.asarray(preds_by_model[name], float)
        fig, axes = plt.subplots(len(areas), 2, figsize=(12, 4.6 * len(areas)), squeeze=False)

        for r, area in enumerate(areas):
            area_mask = meta[area_col].eq(area).to_numpy()
            for c, (seg, flag) in enumerate(SEGMENTS.items()):
                m = meta["inaccessible"].eq(flag).to_numpy() & area_mask
                n_spans = int(m.sum())

                if n_spans < min_n:
                    axes[r, c].axis("off")
                    axes[r, c].set_title(
                        f"{split_label} | {name} | {seg} | {area}\n(n={n_spans} - insufficient data)",
                        fontsize=9)
                    continue

                ta = tier_index(a_all[m], hour_cuts[seg])
                tp = tier_index(p_all[m], hour_cuts[seg])
                plot_tier_confusion(
                    axes[r, c], ta, tp,
                    f"{split_label} | {name} | {seg} | {area}\n"
                    f"hour cuts {list(hour_cuts[seg])} | n={n_spans} | acc {np.mean(ta == tp):.1%}",
                    cmap="Blues" if seg == "Accessible" else "Purples")

        save_path = (save_path_template.format(model=name.lower().replace(" ", "_"))
                     if save_path_template else None)
        finish_figure(fig, save_path, show_plot)
        figs[name] = fig

    return figs

# =============================================================================
# PROTECTIVE DEVICE GROUPING (reporting only - not a model feature, so no
# Train-only fitting rule applies; safe to load and attach at any point,
# including directly onto Test/Holdout predictions).
# =============================================================================

PROTECTIVE_DEVICE_COLS = ["protective_device", "major_protective_device"]
SPAN_DATA_PATH_DEFAULT = "../input_data/span_data.pkl"


def load_protective_device_map(path=SPAN_DATA_PATH_DEFAULT, bay_col="bay",
                                device_cols=PROTECTIVE_DEVICE_COLS) -> pd.DataFrame:
    """
    Load a one-row-per-bay protective-device lookup for reporting.
    ...
    """
    span_data = pd.read_pickle(path)
    if "geometry" in span_data.columns:
        span_data = span_data.drop(columns="geometry")

    missing = [c for c in [bay_col, *device_cols] if c not in span_data.columns]
    if missing:
        raise KeyError(f"span_data is missing expected columns: {missing}")

    def _mode(values: pd.Series) -> str:
        """Self-contained tie-break: sorted first mode value, 'Unknown' if empty.
        Deliberately NOT calling vp.deterministic_mode - that name is not
        confirmed to exist at module level in the current veg_pipeline_updated.py."""
        v = values.dropna().astype(str)
        return "Unknown" if v.empty else sorted(v.mode().tolist())[0]

    raw = span_data[[bay_col, *device_cols]].copy()
    raw[bay_col] = clean_bay_key(raw[bay_col].astype(str))
    raw = raw.dropna(subset=[bay_col])

    rows, multi_bay_counts = [], {c: 0 for c in device_cols}
    for bay_value, group in raw.groupby(bay_col):
        row = {bay_col: bay_value}
        for c in device_cols:
            values = group[c].dropna().astype(str)
            if values.empty:
                row[c] = "Unknown"
                continue
            if values.nunique() > 1:
                multi_bay_counts[c] += 1
            row[c] = _mode(values)   # <-- inlined, not vp.deterministic_mode
        rows.append(row)

    by_bay = pd.DataFrame(rows).set_index(bay_col)

    for c in device_cols:
        n = multi_bay_counts[c]
        if n:
            print(f"NOTE: {n:,} bays map to more than one distinct '{c}' value - "
                  f"each collapsed to a single deterministic value to prevent "
                  f"double-counting in aggregation.")
    print(f"Loaded protective-device map for {len(by_bay):,} bays "
          f"({', '.join(device_cols)}).")
    return by_bay


def attach_device_columns(meta: pd.DataFrame, device_map: pd.DataFrame,
                           bay_col="bay", device_cols=PROTECTIVE_DEVICE_COLS) -> pd.DataFrame:
    """
    Join device_map onto a bay_unique-indexed meta frame via meta['bay'] -
    NOT via meta.index, since meta.index is bay_unique and will not match
    span_data's raw bay id for duplicated spans.
    """
    if bay_col not in meta.columns:
        raise KeyError(f"meta must contain a '{bay_col}' column - rebuild via build_meta().")
    out = meta.copy()
    key = clean_bay_key(out[bay_col].astype(str))
    joined = device_map.reindex(key.to_numpy())
    joined.index = out.index
    for c in device_cols:
        out[c] = joined[c].fillna("Unknown").to_numpy()
    match_rate = float(joined[device_cols[0]].notna().mean())
    print(f"Protective-device match rate onto meta ({len(out):,} rows): {match_rate:.1%}")
    return out


def grouped_actual_vs_predicted(actual, pred, keys) -> pd.DataFrame:
    """Generic version of feeder_table() for an arbitrary grouping key."""
    fr = (pd.DataFrame({"group": keys, "actual": np.asarray(actual, float),
                        "predicted": np.asarray(pred, float)})
          .groupby("group")
          .agg(span_count=("actual", "size"), actual_total_hours=("actual", "sum"),
               predicted_total_hours=("predicted", "sum")))
    fr["error_hours"] = fr["predicted_total_hours"] - fr["actual_total_hours"]
    fr["abs_pct_error"] = fr["error_hours"].abs() / fr["actual_total_hours"].replace(0, np.nan)
    return fr


def grouped_scatter(preds: dict, actual, keys, group_label: str, title_prefix: str,
                     save_path=None, show_plot=False, min_group_n: int = 2) -> dict:
    """
    Generic version of feeder_scatter() for an arbitrary grouping dimension.
    One point per group per model: actual total hours (x) vs predicted total
    hours (y), perfect-prediction diagonal, R^2 annotated in the title -
    same visual language as the existing feeder-level scatter plots.
    """
    n_models = len(preds)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)
    axes = axes[0]
    tables = {}
    for ax, (name, pred) in zip(axes, preds.items()):
        tbl = grouped_actual_vs_predicted(actual, pred, keys)
        tbl = tbl[tbl["span_count"] >= min_group_n]
        tables[name] = tbl
        x, y = tbl["actual_total_hours"].to_numpy(), tbl["predicted_total_hours"].to_numpy()
        r2 = r2_score(x, y) if len(x) > 1 else np.nan
        ax.scatter(x, y, alpha=0.6, s=28)
        lim_max = max(x.max() if len(x) else 1, y.max() if len(y) else 1) * 1.05
        ax.plot([0, lim_max], [0, lim_max], linestyle="--", color="grey", linewidth=1,
                label="Perfect prediction")
        ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
        ax.set_xlabel(f"Actual total manhours per {group_label}")
        ax.set_ylabel(f"Predicted total manhours per {group_label}")
        ax.set_title(f"{title_prefix} | {name}\n({len(tbl):,} {group_label}s, R\u00b2={r2:.3f})")
        ax.legend(loc="upper left", fontsize=8)
    finish_figure(fig, save_path, show_plot)
    return tables


def protective_device_scatter(preds: dict, actual, meta: pd.DataFrame, device_col: str,
                               title_prefix: str, save_path=None, show_plot=False,
                               min_group_n: int = 3) -> dict:
    """Thin wrapper: grouped_scatter() keyed on a protective-device column in meta."""
    if device_col not in meta.columns:
        raise KeyError(f"'{device_col}' not found in meta - call attach_device_columns() first.")
    return grouped_scatter(preds, actual, meta[device_col].to_numpy(),
                           device_col.replace("_", " "), title_prefix,
                           save_path, show_plot, min_group_n)


# =============================================================================
# 16. SERIALISATION + HOLDOUT LOG
# =============================================================================
def jsonable(value):
    if hasattr(value, "to_dict") and not isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict()
    if isinstance(value, ModelUnit):
        return vars(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class HoldoutBreakpoint(Exception):
    """Intentional stop before the final holdout phase."""


def append_holdout_log(path, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(jsonable(record), default=str) + "\n")
