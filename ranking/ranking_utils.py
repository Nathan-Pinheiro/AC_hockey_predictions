"""
Shared helpers for the ranking notebooks (feature analysis, predictions, results).

Goal: predict, the day before a season starts, the final regular-season
ranking of each team inside its conference (2 rankings of ~15 teams per
year), the ranking being given by the points.

Design choices
--------------
* Input is data_prepared/teams.csv (output of data_preparation.ipynb).
  Its numeric columns are MinMax-normalised over the whole table, so raw
  values are recovered from data/Teams.csv for points percentage. Within-season
  z-scores absorb era effects and scale differences.
* The target is the league-wide z-score of the points percentage
  (``Pts / (2 * GP)``) of the season. The predicted ranking is obtained by
  sorting the predicted z-scores inside each (year, conference).
* Every feature only uses information from PREVIOUS seasons (lags). No
  current-season roster, coach or in-season information is used: those
  columns belong to the coach-change task or would leak the target season.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RANKING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RANKING_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DIR = PROJECT_ROOT / "data_prepared"
DATA_PATH = PREPARED_DIR / "teams.csv"
SELECTED_FEATURES_PATH = RANKING_DIR / "selected_features.json"
RESULTS_DIR = RANKING_DIR / "results"

LAGS = 5                 # number of previous seasons used as history
PLAYOFF_SPOTS = 8        # teams qualified per conference (used for a metric)
TEST_START_YEAR = 2005   # everything before is used for design decisions
RANDOM_STATE = 42

REQUIRED_COLUMNS = ["year", "tmID", "franchID", "confID", "Pts", "W", "GF", "GA"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _attach_points_percentage(teams):
    """Add raw points, games and points percentage from data/Teams.csv.

    ``data_prepared/teams.csv`` is MinMax-normalised, so the raw points are not
    recoverable from it.  Points percentage (``Pts / (2 * GP)``) is the fair
    season-level performance measure: it removes the effect of shortened
    seasons (e.g. the 48-game 1995 season) and of unequal games played.
    """
    raw_path = DATA_DIR / "Teams.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found - cannot compute points percentage.")
    raw = pd.read_csv(raw_path)
    raw = raw[raw["lgID"].eq("NHL")].copy()
    raw = raw[["year", "tmID", "G", "Pts", "W"]].rename(columns={
        "G": "games_raw", "Pts": "pts_raw", "W": "wins_raw",
    })
    if raw.duplicated(["year", "tmID"]).any():
        raw = raw.groupby(["year", "tmID"], as_index=False).agg(
            games_raw=("games_raw", "sum"),
            pts_raw=("pts_raw", "sum"),
            wins_raw=("wins_raw", "sum"),
        )
    teams = teams.merge(raw, on=["year", "tmID"], how="left")
    teams["pts_pct"] = teams["pts_raw"] / (2 * teams["games_raw"]).replace(0, np.nan)
    teams["win_pct"] = teams["wins_raw"] / teams["games_raw"].replace(0, np.nan)
    return teams


def load_teams(path=DATA_PATH):
    """Read data_prepared/teams.csv and run basic sanity checks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run data_preparation.ipynb first.")
    teams = pd.read_csv(path)
    if "lgID" in teams.columns:
        teams = teams[teams["lgID"].eq("NHL")]
    missing = [c for c in REQUIRED_COLUMNS if c not in teams.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    if teams.duplicated(["year", "tmID"]).any():
        raise ValueError("Several rows for the same (year, tmID).")
    teams = _attach_points_percentage(teams)
    return teams.reset_index(drop=True)


def _zscore_by_year(df, col):
    grouped = df.groupby("year")[col]
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[col] - grouped.transform("mean")) / std).fillna(0.0)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def build_dataset(teams):
    """
    Returns one row per (team, season) with
      - target       : league-wide z-score of the season points percentage
      - conf_rank    : actual rank in the conference (1 = best), ties broken by
                       wins then team id
      - features     : lagged information only (see feature_groups)

    Points percentage (``Pts / (2 * GP)``) is used instead of raw points or wins:
    it is comparable across the 80-game, 82-game and shortened (48-game) seasons.
    """
    df = teams.dropna(subset=["pts_pct", "GF", "GA"]).copy()
    df["confID"] = df["confID"].fillna("NONE")

    # Season-relative performance (scale free).
    for col in ("pts_pct", "win_pct", "GF", "GA"):
        df[f"{col}_z"] = _zscore_by_year(df, col)
    df["gd_z"] = df["GF_z"] - df["GA_z"]

    # Target ranking inside each conference, based on points percentage.
    df = df.sort_values(
        ["year", "confID", "pts_pct", "wins_raw", "tmID"],
        ascending=[True, True, False, False, True],
    )
    by_conf = df.groupby(["year", "confID"])
    df["conf_rank"] = by_conf.cumcount() + 1
    df["conf_size"] = by_conf["tmID"].transform("size")
    df["conf_rank_pct"] = (df["conf_rank"] - 1) / (df["conf_size"] - 1).clip(lower=1)
    df["target"] = df["pts_pct_z"]

    # History of each franchise.
    df = df.sort_values(["franchID", "year"]).reset_index(drop=True)
    by = df.groupby("franchID", sort=False)

    for k in range(1, LAGS + 1):
        df[f"pts_pct_z_lag{k}"] = by["pts_pct_z"].shift(k)
        df[f"rank_pct_lag{k}"] = by["conf_rank_pct"].shift(k)
    pts_cols = [f"pts_pct_z_lag{k}" for k in range(1, LAGS + 1)]
    rank_cols = [f"rank_pct_lag{k}" for k in range(1, LAGS + 1)]

    df["pts_pct_z_mean2"] = df[pts_cols[:2]].mean(axis=1)
    df["pts_pct_z_mean3"] = df[pts_cols[:3]].mean(axis=1)
    df["pts_pct_z_mean5"] = df[pts_cols].mean(axis=1)
    decay = np.array([0.5 ** k for k in range(LAGS)])
    df["pts_pct_z_ewm"] = (df[pts_cols] * decay).sum(axis=1) / decay.sum()
    df["pts_pct_z_trend"] = df["pts_pct_z_lag1"] - df["pts_pct_z_lag2"]
    df["best_rank_pct_5"] = df[rank_cols].min(axis=1)
    df["worst_rank_pct_5"] = df[rank_cols].max(axis=1)

    df["gf_z_lag1"] = by["GF_z"].shift(1)
    df["ga_z_lag1"] = by["GA_z"].shift(1)
    df["gd_z_lag1"] = by["gd_z"].shift(1)
    df["gd_z_mean3"] = pd.concat(
        [by["gd_z"].shift(k) for k in (1, 2, 3)], axis=1
    ).mean(axis=1)

    if "playoff" in df.columns:
        df["made_playoffs"] = df["playoff"].notna().astype(int)
    else:
        df["made_playoffs"] = 0
    df["made_playoffs_lag1"] = by["made_playoffs"].shift(1)
    df["playoffs_count5"] = by["made_playoffs"].transform(
        lambda s: s.shift(1).rolling(LAGS, min_periods=1).sum()
    )

    df["n_hist"] = df[pts_cols].notna().sum(axis=1)   # 0 = brand-new franchise

    # Neutral values for teams without history (z-score 0 = league average,
    # rank 0.5 = middle of the table); n_hist keeps the information.
    zero_fill = pts_cols + [
        "pts_pct_z_mean3", "pts_pct_z_mean5", "pts_pct_z_trend",
        "gf_z_lag1", "ga_z_lag1", "gd_z_lag1", "gd_z_mean3",
        "made_playoffs_lag1", "playoffs_count5",
    ]
    df[zero_fill] = df[zero_fill].fillna(0.0)
    half_fill = rank_cols + ["best_rank_pct_5", "worst_rank_pct_5"]
    df[half_fill] = df[half_fill].fillna(0.5)

    return df.sort_values(["year", "confID", "conf_rank"]).reset_index(drop=True)


def feature_groups(df):
    """Candidate features, grouped by meaning (used for selection/ablation).

    Only leak-free groups are listed here.  Coach information is deliberately
    excluded: ``coach_changed`` is the target of the coach-change task and
    describes the current season, so using it to predict the standings would
    leak information from the season being predicted.
    """
    groups = {
        "prev_points": ["pts_pct_z_lag1"],
        "history_points": ["pts_pct_z_lag2", "pts_pct_z_lag3", "pts_pct_z_lag4",
                           "pts_pct_z_lag5", "pts_pct_z_mean3", "pts_pct_z_mean5"],
        "trend": ["pts_pct_z_trend"],
        "rank_history": ["rank_pct_lag1", "best_rank_pct_5", "worst_rank_pct_5"],
        "goals": ["gf_z_lag1", "ga_z_lag1", "gd_z_lag1", "gd_z_mean3"],
        "playoffs": ["made_playoffs_lag1", "playoffs_count5"],
        "experience": ["n_hist"],
    }
    return {name: cols for name, cols in groups.items() if cols}


def flatten(groups, names=None):
    names = list(groups) if names is None else names
    return [col for name in names for col in groups[name]]


# ----------------------------------------------------------------------------
# Walk-forward training / evaluation
# ----------------------------------------------------------------------------
def walk_forward(df, model_factory, features, eval_years, min_hist=1):
    """
    For each evaluation year Y: fit on all seasons < Y, predict Y.
    This mimics 'the day before the season starts' and never looks ahead.
    """
    outputs = []
    for year in eval_years:
        train = df[(df["year"] < year) & (df["n_hist"] >= min_hist)]
        test = df[df["year"] == year]
        if test.empty or len(train) < 50:
            continue
        model = model_factory()
        model.fit(train[features], train["target"])
        out = test[["year", "tmID", "confID"]].copy()
        out["score"] = model.predict(test[features])
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def baseline_scores(df, column, eval_years):
    """Naive model: the score is directly one of the history columns."""
    out = df.loc[df["year"].isin(list(eval_years)), ["year", "tmID", "confID"]].copy()
    out["score"] = df.loc[out.index, column]
    return out.reset_index(drop=True)


def rank_predictions(pred, df):
    """Turn scores into predicted ranks inside each (year, conference)."""
    actual = df[["year", "tmID", "conf_rank", "conf_size", "target"]]
    d = pred.merge(actual, on=["year", "tmID"], how="inner")
    d = d[d["confID"].ne("NONE")]
    d = d.sort_values(["year", "confID", "score", "tmID"],
                      ascending=[True, True, False, True])
    d["pred_rank"] = d.groupby(["year", "confID"]).cumcount() + 1
    d["err"] = d["pred_rank"] - d["conf_rank"]
    d["abs_err"] = d["err"].abs()
    return d.reset_index(drop=True)


def season_metrics(ranked):
    """One row per (year, conference)."""
    rows = []
    for (year, conf), g in ranked.groupby(["year", "confID"]):
        n_top = min(PLAYOFF_SPOTS, len(g) // 2)
        pred_top = set(g.loc[g["pred_rank"] <= n_top, "tmID"])
        true_top = set(g.loc[g["conf_rank"] <= n_top, "tmID"])
        rows.append({
            "year": year,
            "confID": conf,
            "n_teams": len(g),
            "mae": g["abs_err"].mean(),
            "rmse": float(np.sqrt((g["err"] ** 2).mean())),
            "spearman": g["pred_rank"].corr(g["conf_rank"], method="spearman"),
            "top_overlap": len(pred_top & true_top) / n_top,
            "within_2": (g["abs_err"] <= 2).mean(),
        })
    return pd.DataFrame(rows)


def summarise(season_table):
    return season_table[["mae", "rmse", "spearman", "top_overlap", "within_2"]].mean()


def random_ranking_mae(n):
    """Expected MAE of a uniformly random permutation of n teams."""
    return (n ** 2 - 1) / (3 * n)


def paired_bootstrap(err_a, err_b, groups, n_boot=2000, seed=RANDOM_STATE):
    """
    Mean of (err_a - err_b) with a 95% CI obtained by resampling whole
    (year, conference) groups. A positive mean means model B is better.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"d": np.asarray(err_a) - np.asarray(err_b), "g": np.asarray(groups)})
    agg = frame.groupby("g")["d"].agg(["sum", "count"])
    sums, counts = agg["sum"].to_numpy(), agg["count"].to_numpy()
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return sums.sum() / counts.sum(), tuple(np.percentile(means, [2.5, 97.5]))
