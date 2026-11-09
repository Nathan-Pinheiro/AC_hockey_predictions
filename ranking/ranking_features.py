"""External ranking features built from the raw hockey tables.

The base dataset (``data_prepared/teams.csv``) only contains season-level team
stats.  This module enriches it with information that is available before the
season starts:

* shots, shooting %, save %, PDO and shot differential (Scoring / Goalies / Teams)
* special teams: power play, penalty kill, penalties (Teams)
* goalie quality and workload (Goalies)
* playoff history (SeriesPost / TeamsPost)
* player awards won by the previous roster (AwardsPlayers)
* home/road and second-half performance (TeamSplits)

Leak policy
-----------
Every block is lagged: the value of season Y-1 predicts season Y.  Current-season
roster composition and coach information are deliberately **excluded**: the
roster is only known after the season (in-season arrivals), and coach-change
columns are the target of the separate coach task.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _z(df, col):
    g = df.groupby("year")[col]
    std = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / std).fillna(0.0)


def _next_season_map(years):
    years = sorted(set(int(y) for y in years))
    return dict(zip(years[:-1], years[1:]))


def _lag(df, cols, keys=("year", "tmID")):
    """Attach next-season values: the value of season Y-1 becomes a feature for Y."""
    nxt = _next_season_map(df["year"].unique())
    prev = df.copy()
    prev["year"] = prev["year"].map(nxt)
    prev = prev.dropna(subset=["year"]).astype({"year": int})
    prev = prev[list(keys) + list(cols)].rename(columns={c: c + "_lag1" for c in cols})
    return df.merge(prev, on=list(keys), how="left")


def build_external_features(data_dir=None):
    """Return a ``(features, groups)`` pair for the external feature blocks.

    ``features`` has one row per ``(year, tmID)`` and ``groups`` maps a readable
    block name to the list of column names it contains.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    # ---------------------------------------------------------------- teams
    teams = pd.read_csv(data_dir / "Teams.csv")
    teams = teams[teams["lgID"].eq("NHL")].copy()
    teams = _num(teams, ["G", "W", "L", "T", "OTL", "Pts", "GF", "GA", "PIM",
                         "PPG", "PPC", "PKG", "PKC", "SHF"])

    # ---------------------------------------------------------------- shots
    scoring = pd.read_csv(data_dir / "Scoring.csv")
    scoring = scoring[scoring["lgID"].eq("NHL")].copy()
    scoring = _num(scoring, ["GP", "G", "A", "Pts", "SOG", "PPG", "PPA", "SHG"])
    scoring["pos"] = scoring["pos"].fillna("")
    skaters = scoring[~scoring["pos"].eq("G")].copy()
    shots = skaters.groupby(["year", "tmID"], as_index=False)["SOG"].sum().rename(columns={"SOG": "SF"})

    # ---------------------------------------------------------------- goalies
    goalies = pd.read_csv(data_dir / "Goalies.csv")
    goalies = goalies[goalies["lgID"].eq("NHL")].copy()
    goalies = _num(goalies, ["GP", "Min", "W", "L", "GA", "SA"])
    gteam = goalies.groupby(["year", "tmID"], as_index=False)[["Min", "SA", "GA"]].sum()
    gteam = gteam.rename(columns={"GA": "GA_G"})
    gteam["team_sv"] = 1 - gteam["GA_G"] / gteam["SA"].replace(0, np.nan)

    # main goalie of each team-season (by minutes)
    gp = goalies.groupby(["playerID", "year", "tmID"], as_index=False)[["Min", "SA", "GA"]].sum()
    main = (gp.sort_values(["year", "tmID", "Min"], ascending=[True, True, False])
              .drop_duplicates(["year", "tmID"]))
    main = main.rename(columns={"Min": "starter_min", "SA": "starter_SA", "GA": "starter_GA"})

    # ---------------------------------------------------------------- team table
    ext = teams[["year", "tmID", "franchID", "G", "GF", "GA", "PIM",
                 "PPG", "PPC", "PKG", "PKC"]].merge(shots, on=["year", "tmID"], how="left")
    ext = ext.merge(gteam[["year", "tmID", "SA", "GA_G", "team_sv"]], on=["year", "tmID"], how="left")
    ext = ext.merge(main[["year", "tmID", "starter_min", "starter_SA", "starter_GA"]],
                    on=["year", "tmID"], how="left")

    ext["sh_pct"] = ext["GF"] / ext["SF"].replace(0, np.nan)
    ext["sv_pct"] = 1 - ext["GA_G"] / ext["SA"].replace(0, np.nan)
    ext["pdo"] = ext["sh_pct"] + ext["sv_pct"]
    ext["shot_diff"] = ext["SF"] - ext["SA"]
    ext["shot_diff_pct"] = ext["shot_diff"] / (ext["SF"] + ext["SA"]).replace(0, np.nan)
    ext["pp_pct"] = ext["PPG"] / ext["PPC"].replace(0, np.nan)
    ext["pk_pct"] = 1 - ext["PKG"] / ext["PKC"].replace(0, np.nan)
    ext["pim_per_game"] = ext["PIM"] / ext["G"].replace(0, np.nan)
    ext["starter_sv"] = 1 - ext["starter_GA"] / ext["starter_SA"].replace(0, np.nan)

    # lagged team-performance features (known before the season)
    lag_cols = ["sh_pct", "sv_pct", "pdo", "shot_diff", "shot_diff_pct",
                "pp_pct", "pk_pct", "pim_per_game", "team_sv", "starter_sv", "starter_min"]
    ext = _lag(ext, lag_cols)

    # 3-season means for the noisiest rates
    ext = ext.sort_values(["franchID", "year"]).reset_index(drop=True)
    by = ext.groupby("franchID", sort=False)
    for c in ["pdo", "shot_diff_pct", "team_sv", "starter_sv"]:
        ext[c + "_mean3"] = pd.concat([by[c].shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)

    # z-scores by season
    for c in lag_cols + [c + "_mean3" for c in ["pdo", "shot_diff_pct", "team_sv", "starter_sv"]]:
        ext[c + "_z"] = _z(ext, c)

    # ---------------------------------------------------------------- previous roster
    # Only used to map awards won in season Y-1 back to the team the player
    # played for in Y-1.  No current-season roster information is used.
    player_team = skaters.groupby(["playerID", "year", "tmID"], as_index=False)["Pts"].sum()
    nxt = _next_season_map(player_team["year"].unique())

    # ---------------------------------------------------------------- playoff history
    series = pd.read_csv(data_dir / "SeriesPost.csv")
    series = series[series["lgIDWinner"].eq("NHL") | series["lgIDLoser"].eq("NHL")].copy()
    wins = series.groupby(["year", "tmIDWinner"]).size().rename("playoff_rounds_won").reset_index()
    wins = wins.rename(columns={"tmIDWinner": "tmID"})
    post = pd.read_csv(data_dir / "TeamsPost.csv")
    post = post[post["lgID"].eq("NHL")].copy()
    post = _num(post, ["G", "W", "L", "T"])
    post["playoff_win_pct"] = post["W"] / post["G"].replace(0, np.nan)
    post = post[["year", "tmID", "G", "W", "playoff_win_pct"]].rename(
        columns={"G": "playoff_games", "W": "playoff_wins"})
    ph = post.merge(wins, on=["year", "tmID"], how="left")
    ph["playoff_rounds_won"] = ph["playoff_rounds_won"].fillna(0)
    ext = ext.merge(ph, on=["year", "tmID"], how="left")
    for c in ["playoff_rounds_won", "playoff_games", "playoff_wins", "playoff_win_pct"]:
        ext[c + "_z"] = _z(ext, c)

    # ---------------------------------------------------------------- awards (previous roster)
    awards = pd.read_csv(data_dir / "AwardsPlayers.csv")
    awards = awards[awards["lgID"].eq("NHL")].copy()
    awards["is_all_star"] = awards["award"].isin(["First Team All-Star", "Second Team All-Star"])
    awards["is_major"] = awards["award"].isin(["Hart", "Norris", "Vezina", "Art Ross", "Calder", "Selke", "Pearson"])
    award_count = awards.groupby(["year", "playerID"]).agg(
        award_count=("award", "size"),
        all_star_count=("is_all_star", "sum"),
        major_award_count=("is_major", "sum"),
    ).reset_index()
    prev_roster = player_team[["playerID", "year", "tmID"]].merge(
        award_count, on=["playerID", "year"], how="left")
    prev_roster["year"] = prev_roster["year"].map(nxt)
    prev_roster = prev_roster.dropna(subset=["year"]).astype({"year": int})
    aw = prev_roster.groupby(["year", "tmID"], as_index=False)[
        ["award_count", "all_star_count", "major_award_count"]].sum()
    ext = ext.merge(aw, on=["year", "tmID"], how="left")
    for c in ["award_count", "all_star_count", "major_award_count"]:
        ext[c] = ext[c].fillna(0.0)
        ext[c + "_z"] = _z(ext, c)

    # ---------------------------------------------------------------- splits
    splits = pd.read_csv(data_dir / "TeamSplits.csv")
    splits = splits[splits["lgID"].eq("NHL")].copy()
    splits = _num(splits, [c for c in splits.columns if c not in ("year", "lgID", "tmID")])
    def rate(w, l, t, otl):
        return w / (w + l + t + otl).replace(0, np.nan)
    splits["home_win_pct"] = rate(splits["hW"], splits["hL"], splits["hT"], splits["hOTL"])
    splits["road_win_pct"] = rate(splits["rW"], splits["rL"], splits["rT"], splits["rOTL"])
    second = ["Jan", "Feb", "Mar", "Apr"]
    w = sum(splits[f"{m}W"] for m in second)
    l = sum(splits[f"{m}L"] for m in second)
    t = sum(splits[f"{m}T"] for m in second)
    otl = sum(splits[f"{m}OL"] for m in second)
    splits["second_half_win_pct"] = w / (w + l + t + otl).replace(0, np.nan)
    ext = ext.merge(splits[["year", "tmID", "home_win_pct", "road_win_pct", "second_half_win_pct"]],
                    on=["year", "tmID"], how="left")
    for c in ["home_win_pct", "road_win_pct", "second_half_win_pct"]:
        ext[c + "_z"] = _z(ext, c)

    # ---------------------------------------------------------------- final columns
    keep = ["year", "tmID", "franchID"]
    # Columns that describe the *current* season and must be lagged by one season.
    lag_now = {
        "shots_luck": ["sh_pct", "sv_pct", "pdo", "shot_diff", "shot_diff_pct"],
        "special_teams": ["pp_pct", "pk_pct", "pim_per_game"],
        "goalie": ["team_sv", "starter_sv", "starter_min"],
        "playoffs": ["playoff_rounds_won", "playoff_games", "playoff_wins", "playoff_win_pct"],
        "splits": ["home_win_pct", "road_win_pct", "second_half_win_pct"],
    }
    for cols in lag_now.values():
        for c in cols:
            ext[c + "_lag1"] = ext.groupby("franchID")[c].shift(1)
            ext[c + "_lag1_z"] = _z(ext, c + "_lag1")

    out_groups = {
        "shots_luck": [c + "_lag1_z" for c in lag_now["shots_luck"]] +
                      ["pdo_mean3_z", "shot_diff_pct_mean3_z"],
        "special_teams": [c + "_lag1_z" for c in lag_now["special_teams"]],
        "goalie": ["team_sv_lag1_z", "team_sv_mean3_z",
                   "starter_sv_lag1_z", "starter_sv_mean3_z", "starter_min_lag1_z"],
        "playoff_history": [c + "_lag1_z" for c in lag_now["playoffs"]],
        "awards": ["award_count_z", "all_star_count_z", "major_award_count_z"],
        "splits": [c + "_lag1_z" for c in lag_now["splits"]],
    }
    cols = keep + [c for g in out_groups.values() for c in g]
    out = ext[cols].copy()
    for c in cols[3:]:
        out[c] = out[c].fillna(0.0)
    return out, out_groups
