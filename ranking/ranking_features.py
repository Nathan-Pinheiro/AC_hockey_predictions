"""External ranking features built from the raw hockey tables.

The base dataset (``data_prepared/teams.csv``) only contains season-level team
stats.  This module enriches it with information that is available before the
season starts.

Feature blocks
--------------
* ``shots_luck``      : shooting %, save %, PDO, shot differential.
* ``special_teams``   : power play, penalty kill, penalties.
* ``goalie``          : team / starter save % and starter workload.
* ``splits``          : home / road / second-half / spring points percentage.
* ``teamvs``          : results against playoff teams, top-8 teams, opponent
                        strength and same-conference opponents.
* ``roster``          : lagged roster composition (scoring depth, top-6
                        concentration, age, continuity) and goalie depth.
* ``awards``          : awards won by the previous season's roster.
* ``playoff_history`` : playoff rounds / games / wins of the previous season.

Leak policy
-----------
Every block is **lagged**: the value of season Y-1 predicts season Y.  The
roster block uses the players who played for the team in Y-1, which is fully
known before Y starts.  Current-season roster composition, current-season
``TeamsPost`` / ``SeriesPost`` and coach columns are deliberately excluded:
the roster / playoff result is only known after the season, and coach-change
columns are the target of the separate coach task.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Raw metrics that are lagged by one season inside each franchise.
LAG_METRICS = [
    # shots / luck
    "sh_pct", "sv_pct", "pdo", "shot_diff", "shot_diff_pct",
    # special teams
    "pp_pct", "pk_pct", "pim_per_game",
    # goalies
    "team_sv", "starter_sv", "starter_min",
    # splits
    "home_pts_pct", "road_pts_pct", "delta_home_road",
    "second_half_pts_pct", "spring_pts_pct",
    # head-to-head / strength of schedule
    "winrate_vs_playoff", "winrate_vs_top8",
    "weighted_opp_strength", "winrate_vs_current_conf",
    # roster composition (season Y-1)
    "n_20goal", "n_ppg", "top6_pts_share", "avg_age",
    "roster_continuity",
    # goalie depth (season Y-1)
    "starter_save_pct", "starter_gp", "backup_save_pct",
    # playoff history (season Y-1)
    "playoff_rounds_won", "playoff_games", "playoff_wins", "playoff_win_pct",
    # awards (already mapped from the Y-1 roster)
    "award_count", "all_star_count", "major_award_count",
]

MEAN3_METRICS = ["pdo", "shot_diff_pct", "team_sv", "starter_sv"]


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _z(df, col):
    g = df.groupby("year")[col]
    std = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / std).fillna(0.0)


def _safe_div(a, b):
    if np.isscalar(b):
        return a / b if b else np.nan
    return a / b.replace(0, np.nan)


def _next_season_map(years):
    years = sorted(set(int(y) for y in years))
    return dict(zip(years[:-1], years[1:]))


def _prev_season_map(years):
    years = sorted(set(int(y) for y in years))
    return {y: years[i - 1] for i, y in enumerate(years) if i > 0}


def _pts_pct(w, l, t, otl):
    """Points percentage from W/L/T/OTL counts (OTL gives one point)."""
    w = w.fillna(0)
    l = l.fillna(0)
    t = t.fillna(0)
    otl = otl.fillna(0)
    games = w + l + t + otl
    points = 2 * w + t + otl
    return _safe_div(points, 2 * games)


def _lag_by_franchise(ext, cols):
    """Add ``{col}_lag1`` = value of the previous season inside each franchise.

    The table is sorted by ``(franchID, year)``; the missing 2004 season is
    handled naturally by the sort (2003 is followed by 2005).
    """
    ext = ext.sort_values(["franchID", "year"]).reset_index(drop=True)
    by = ext.groupby("franchID", sort=False)
    for c in cols:
        ext[c + "_lag1"] = by[c].shift(1)
    return ext


# ---------------------------------------------------------------------------
# Roster helpers
# ---------------------------------------------------------------------------
def _roster_metrics(skaters, teams, master):
    """Lagged roster composition, computed per (season, franchise)."""
    # Collapse stints: one row per (player, season, team).
    pt = skaters.groupby(["playerID", "year", "tmID"], as_index=False).agg(
        GP=("GP", "sum"), G=("G", "sum"), Pts=("Pts", "sum"),
    )
    # Follow the franchise, not the unstable tmID.
    pt = pt.merge(teams[["year", "tmID", "franchID"]], on=["year", "tmID"], how="left")
    pf = pt.groupby(["playerID", "year", "franchID"], as_index=False).agg(
        GP=("GP", "sum"), G=("G", "sum"), Pts=("Pts", "sum"),
    )

    # Age from Master.
    pf = pf.merge(master[["playerID", "birthYear"]], on="playerID", how="left")
    pf["age"] = pf["year"] - pf["birthYear"]

    # Continuity: was the player already in the same franchise one season before?
    prev = _prev_season_map(pf["year"].unique())
    presence = set(zip(pf["playerID"], pf["year"], pf["franchID"]))
    pf["was_here"] = [
        int((p, prev.get(y), f) in presence)
        for p, y, f in zip(pf["playerID"], pf["year"], pf["franchID"])
    ]

    def agg(g):
        gp = g["GP"].sum()
        pts = g["Pts"].sum()
        ages = g["age"]
        if ages.notna().any():
            avg_age = float(np.average(ages.fillna(ages.mean()),
                                       weights=g["GP"].clip(lower=1)))
        else:
            avg_age = np.nan
        return pd.Series({
            "n_20goal": int((g["G"] >= 20).sum()),
            "n_ppg": int(((g["GP"] >= 20) &
                          (_safe_div(g["Pts"], g["GP"]) >= 1.0)).sum()),
            "top6_pts_share": (_safe_div(g.nlargest(6, "Pts")["Pts"].sum(), pts)
                               if pts > 0 else np.nan),
            "avg_age": avg_age,
            "roster_continuity": (_safe_div((g["GP"] * g["was_here"]).sum(), gp)
                                  if gp > 0 else np.nan),
        })

    return pf.groupby(["year", "franchID"]).apply(agg).reset_index()


def _goalie_depth(goalies, teams):
    """Starter / backup save percentage and starter workload per franchise-season."""
    gp = goalies.groupby(["playerID", "year", "tmID"], as_index=False).agg(
        GP=("GP", "sum"), Min=("Min", "sum"), SA=("SA", "sum"), GA=("GA", "sum"),
    )
    gp = gp.merge(teams[["year", "tmID", "franchID"]], on=["year", "tmID"], how="left")
    pf = gp.groupby(["playerID", "year", "franchID"], as_index=False).agg(
        GP=("GP", "sum"), Min=("Min", "sum"), SA=("SA", "sum"), GA=("GA", "sum"),
    )
    pf["sv_pct"] = 1 - _safe_div(pf["GA"], pf["SA"])

    pf = pf.sort_values(["year", "franchID", "GP"], ascending=[True, True, False])
    pf["depth_rank"] = pf.groupby(["year", "franchID"]).cumcount() + 1

    starter = pf[pf["depth_rank"] == 1][["year", "franchID", "sv_pct", "GP"]].rename(
        columns={"sv_pct": "starter_save_pct", "GP": "starter_gp"})
    backup = pf[pf["depth_rank"] == 2][["year", "franchID", "sv_pct"]].rename(
        columns={"sv_pct": "backup_save_pct"})
    return starter.merge(backup, on=["year", "franchID"], how="outer")


def _team_vs_team(teamvs, teams):
    """Strength-of-schedule / head-to-head features from TeamVsTeam."""
    vt = teamvs.copy()
    vt = _num(vt, ["W", "L", "T", "OTL"])
    for c in ["W", "L", "T", "OTL"]:
        vt[c] = vt[c].fillna(0)
    vt["games"] = vt["W"] + vt["L"] + vt["T"] + vt["OTL"]

    own = teams[["year", "tmID", "confID"]].rename(columns={"confID": "own_confID"})
    opp = teams[["year", "tmID", "confID", "pts_pct", "playoff_team", "top8"]].rename(
        columns={"tmID": "oppID", "confID": "opp_confID", "pts_pct": "opp_pts_pct",
                 "playoff_team": "opp_playoff", "top8": "opp_top8"})
    vt = vt.merge(own, on=["year", "tmID"], how="left")
    vt = vt.merge(opp, on=["year", "oppID"], how="left")

    def agg(g):
        games = g["games"].sum()
        if games <= 0:
            return pd.Series({
                "winrate_vs_playoff": np.nan, "winrate_vs_top8": np.nan,
                "weighted_opp_strength": np.nan,
                "winrate_vs_current_conf": np.nan,
            })
        playoff = g[g["opp_playoff"].eq(1)]
        top8 = g[g["opp_top8"].eq(1)]
        same_conf = g[g["own_confID"].eq(g["opp_confID"])]
        return pd.Series({
            "winrate_vs_playoff": _safe_div(playoff["W"].sum(), playoff["games"].sum()),
            "winrate_vs_top8": _safe_div(top8["W"].sum(), top8["games"].sum()),
            "weighted_opp_strength": _safe_div(
                (g["opp_pts_pct"].fillna(0) * g["games"]).sum(), games),
            "winrate_vs_current_conf": _safe_div(
                same_conf["W"].sum(), same_conf["games"].sum()),
        })

    return vt.groupby(["year", "tmID"]).apply(agg).reset_index()


def _playoff_history(series, post, teams):
    """Playoff rounds / games / wins of a season, per franchise."""
    wins = series.groupby(["year", "tmIDWinner"]).size().rename(
        "playoff_rounds_won").reset_index().rename(columns={"tmIDWinner": "tmID"})
    ph = post[["year", "tmID", "G", "W"]].copy()
    ph["playoff_win_pct"] = _safe_div(ph["W"], ph["G"])
    ph = ph.rename(columns={"G": "playoff_games", "W": "playoff_wins"})
    ph = ph.merge(wins, on=["year", "tmID"], how="left")
    ph["playoff_rounds_won"] = ph["playoff_rounds_won"].fillna(0)
    ph = ph.merge(teams[["year", "tmID", "franchID"]], on=["year", "tmID"], how="left")
    return ph.groupby(["year", "franchID"], as_index=False)[
        ["playoff_rounds_won", "playoff_games", "playoff_wins", "playoff_win_pct"]
    ].sum(min_count=1)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_external_features(data_dir=None):
    """Return a ``(features, groups)`` pair for the external feature blocks.

    ``features`` has one row per ``(year, tmID)`` and ``groups`` maps a readable
    block name to the list of column names it contains.  All feature columns
    are z-scored by season and lagged (season Y-1 -> season Y).
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    # ---------------------------------------------------------------- teams
    teams = pd.read_csv(data_dir / "Teams.csv")
    teams = teams[teams["lgID"].eq("NHL")].copy()
    teams = _num(teams, ["G", "W", "L", "T", "OTL", "Pts", "GF", "GA", "PIM",
                         "PPG", "PPC", "PKG", "PKC", "SHF"])
    teams["pts_pct"] = _safe_div(teams["Pts"], 2 * teams["G"])
    teams["playoff_team"] = teams["playoff"].notna().astype(int)
    teams = teams.sort_values(["year", "pts_pct", "tmID"],
                              ascending=[True, False, True])
    teams["league_rank"] = teams.groupby("year").cumcount() + 1
    teams["top8"] = (teams["league_rank"] <= 8).astype(int)
    teams = teams.sort_values(["franchID", "year"]).reset_index(drop=True)

    # ---------------------------------------------------------------- shots
    scoring = pd.read_csv(data_dir / "Scoring.csv")
    scoring = scoring[scoring["lgID"].eq("NHL")].copy()
    scoring = _num(scoring, ["GP", "G", "A", "Pts", "SOG", "PPG", "PPA", "SHG"])
    scoring["pos"] = scoring["pos"].fillna("")
    skaters = scoring[~scoring["pos"].eq("G")].copy()
    shots = skaters.groupby(["year", "tmID"], as_index=False)["SOG"].sum().rename(
        columns={"SOG": "SF"})

    # ---------------------------------------------------------------- goalies
    goalies = pd.read_csv(data_dir / "Goalies.csv")
    goalies = goalies[goalies["lgID"].eq("NHL")].copy()
    goalies = _num(goalies, ["GP", "Min", "W", "L", "GA", "SA"])
    gteam = goalies.groupby(["year", "tmID"], as_index=False)[["Min", "SA", "GA"]].sum()
    gteam = gteam.rename(columns={"GA": "GA_G"})
    gteam["team_sv"] = 1 - _safe_div(gteam["GA_G"], gteam["SA"])

    gp = goalies.groupby(["playerID", "year", "tmID"], as_index=False)[
        ["Min", "SA", "GA"]].sum()
    main = (gp.sort_values(["year", "tmID", "Min"], ascending=[True, True, False])
              .drop_duplicates(["year", "tmID"]))
    main = main.rename(columns={"Min": "starter_min", "SA": "starter_SA",
                                "GA": "starter_GA"})

    # ---------------------------------------------------------------- base table
    ext = teams[["year", "tmID", "franchID", "pts_pct", "G", "GF", "GA", "PIM",
                 "PPG", "PPC", "PKG", "PKC"]].merge(
        shots, on=["year", "tmID"], how="left")
    ext = ext.merge(gteam[["year", "tmID", "SA", "GA_G", "team_sv"]],
                    on=["year", "tmID"], how="left")
    ext = ext.merge(main[["year", "tmID", "starter_min", "starter_SA", "starter_GA"]],
                    on=["year", "tmID"], how="left")

    ext["sh_pct"] = _safe_div(ext["GF"], ext["SF"])
    ext["sv_pct"] = 1 - _safe_div(ext["GA_G"], ext["SA"])
    ext["pdo"] = ext["sh_pct"] + ext["sv_pct"]
    ext["shot_diff"] = ext["SF"] - ext["SA"]
    ext["shot_diff_pct"] = _safe_div(ext["shot_diff"], ext["SF"] + ext["SA"])
    ext["pp_pct"] = _safe_div(ext["PPG"], ext["PPC"])
    ext["pk_pct"] = 1 - _safe_div(ext["PKG"], ext["PKC"])
    ext["pim_per_game"] = _safe_div(ext["PIM"], ext["G"])
    ext["starter_sv"] = 1 - _safe_div(ext["starter_GA"], ext["starter_SA"])

    # ---------------------------------------------------------------- splits
    splits = pd.read_csv(data_dir / "TeamSplits.csv")
    splits = splits[splits["lgID"].eq("NHL")].copy()
    splits = _num(splits, [c for c in splits.columns
                           if c not in ("year", "lgID", "tmID")])
    splits["home_pts_pct"] = _pts_pct(splits["hW"], splits["hL"],
                                      splits["hT"], splits["hOTL"])
    splits["road_pts_pct"] = _pts_pct(splits["rW"], splits["rL"],
                                      splits["rT"], splits["rOTL"])
    splits["delta_home_road"] = splits["home_pts_pct"] - splits["road_pts_pct"]

    second = ["Jan", "Feb", "Mar", "Apr"]
    spring = ["Mar", "Apr"]

    def _sum_months(months, suffix):
        cols = [f"{m}{suffix}" for m in months if f"{m}{suffix}" in splits.columns]
        return splits[cols].fillna(0).sum(axis=1)

    splits["second_half_pts_pct"] = _pts_pct(
        _sum_months(second, "W"), _sum_months(second, "L"),
        _sum_months(second, "T"), _sum_months(second, "OL"))
    splits["spring_pts_pct"] = _pts_pct(
        _sum_months(spring, "W"), _sum_months(spring, "L"),
        _sum_months(spring, "T"), _sum_months(spring, "OL"))
    ext = ext.merge(
        splits[["year", "tmID", "home_pts_pct", "road_pts_pct",
                "delta_home_road", "second_half_pts_pct", "spring_pts_pct"]],
        on=["year", "tmID"], how="left")

    # ---------------------------------------------------------------- head-to-head
    teamvs = pd.read_csv(data_dir / "TeamVsTeam.csv")
    teamvs = teamvs[teamvs["lgID"].eq("NHL")].copy()
    vt = _team_vs_team(teamvs, teams)
    ext = ext.merge(vt, on=["year", "tmID"], how="left")

    # ---------------------------------------------------------------- roster
    master = pd.read_csv(data_dir / "Master.csv")
    roster = _roster_metrics(skaters, teams, master)
    ext = ext.merge(roster, on=["year", "franchID"], how="left")

    depth = _goalie_depth(goalies, teams)
    ext = ext.merge(depth, on=["year", "franchID"], how="left")

    # ---------------------------------------------------------------- playoffs
    series = pd.read_csv(data_dir / "SeriesPost.csv")
    series = series[series["lgIDWinner"].eq("NHL") | series["lgIDLoser"].eq("NHL")].copy()
    series = _num(series, ["W", "L", "T"])
    post = pd.read_csv(data_dir / "TeamsPost.csv")
    post = post[post["lgID"].eq("NHL")].copy()
    post = _num(post, ["G", "W", "L", "T"])
    ph = _playoff_history(series, post, teams)
    ext = ext.merge(ph, on=["year", "franchID"], how="left")

    # ---------------------------------------------------------------- awards
    awards = pd.read_csv(data_dir / "AwardsPlayers.csv")
    awards = awards[awards["lgID"].eq("NHL")].copy()
    awards["is_all_star"] = awards["award"].isin(
        ["First Team All-Star", "Second Team All-Star"])
    awards["is_major"] = awards["award"].isin(
        ["Hart", "Norris", "Vezina", "Art Ross", "Calder", "Selke", "Pearson"])
    award_count = awards.groupby(["year", "playerID"]).agg(
        award_count=("award", "size"),
        all_star_count=("is_all_star", "sum"),
        major_award_count=("is_major", "sum"),
    ).reset_index()

    # Map awards won in season S to the team the player played for in S;
    # the feature is then lagged to S+1 together with the other metrics.
    player_team = skaters.groupby(["playerID", "year", "tmID"], as_index=False)[
        "Pts"].sum()
    player_team = player_team.merge(teams[["year", "tmID", "franchID"]],
                                    on=["year", "tmID"], how="left")
    player_fr = player_team.groupby(["playerID", "year", "franchID"],
                                    as_index=False)["Pts"].sum()
    aw = player_fr.merge(award_count, on=["playerID", "year"], how="left")
    aw = aw.groupby(["year", "franchID"], as_index=False)[
        ["award_count", "all_star_count", "major_award_count"]].sum()
    ext = ext.merge(aw, on=["year", "franchID"], how="left")

    # ---------------------------------------------------------------- lag
    ext = _lag_by_franchise(ext, LAG_METRICS)

    # 3-season means for the noisiest rates (already lagged by construction).
    by = ext.groupby("franchID", sort=False)
    for c in MEAN3_METRICS:
        ext[c + "_mean3"] = pd.concat(
            [by[c].shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)

    lag_cols = [c + "_lag1" for c in LAG_METRICS]
    mean3_cols = [c + "_mean3" for c in MEAN3_METRICS]
    for c in lag_cols + mean3_cols:
        ext[c + "_z"] = _z(ext, c)

    # ---------------------------------------------------------------- groups
    out_groups = {
        "shots_luck": [
            "sh_pct_lag1_z", "sv_pct_lag1_z", "pdo_lag1_z",
            "shot_diff_lag1_z", "shot_diff_pct_lag1_z",
            "pdo_mean3_z", "shot_diff_pct_mean3_z",
        ],
        "special_teams": [
            "pp_pct_lag1_z", "pk_pct_lag1_z", "pim_per_game_lag1_z",
        ],
        "goalie": [
            "team_sv_lag1_z", "team_sv_mean3_z",
            "starter_sv_lag1_z", "starter_sv_mean3_z", "starter_min_lag1_z",
        ],
        "splits": [
            "home_pts_pct_lag1_z", "road_pts_pct_lag1_z",
            "delta_home_road_lag1_z", "second_half_pts_pct_lag1_z",
            "spring_pts_pct_lag1_z",
        ],
        "teamvs": [
            "winrate_vs_playoff_lag1_z", "winrate_vs_top8_lag1_z",
            "weighted_opp_strength_lag1_z", "winrate_vs_current_conf_lag1_z",
        ],
        "roster": [
            "n_20goal_lag1_z", "n_ppg_lag1_z", "top6_pts_share_lag1_z",
            "avg_age_lag1_z", "roster_continuity_lag1_z",
            "starter_save_pct_lag1_z", "starter_gp_lag1_z",
            "backup_save_pct_lag1_z",
        ],
        "awards": [
            "award_count_lag1_z", "all_star_count_lag1_z",
            "major_award_count_lag1_z",
        ],
        "playoff_history": [
            "playoff_rounds_won_lag1_z", "playoff_games_lag1_z",
            "playoff_wins_lag1_z", "playoff_win_pct_lag1_z",
        ],
    }
    keep = ["year", "tmID", "franchID"]
    cols = keep + [c for g in out_groups.values() for c in g]
    out = ext[cols].copy()
    for c in cols[3:]:
        out[c] = out[c].fillna(0.0)
    return out, out_groups
