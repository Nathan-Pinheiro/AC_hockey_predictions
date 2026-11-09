"""
Roster-level features built from the RAW player tables (data/Scoring.csv, data/Goalies.csv).

Raw tables are used (not data_prepared) because per-player values must be summed per team and
the prepared files are MinMax-normalised.

Everything about a player's TALENT comes from the PREVIOUS season (never the season to predict).
The only information taken from the predicted season is WHO is on the team roster.
This is an approximation of "the roster known the day before the season": players
traded or signed during the season are included (mild leak, see the notebook).
"""
import numpy as np
import pandas as pd

GOALIE_PRIOR_SHOTS = 600   # shrinkage of the previous-season save percentage toward the league mean

ROSTER_GROUPS = {
    "roster_talent": ["roster_prev_pts_z", "top5_prev_pts_z"],
    "talent_moves": ["talent_added_z", "talent_lost_z", "net_talent_z"],
    "continuity": ["continuity_z"],
    "goalie": ["goalie_prev_sv_z"],
}


def _z_by_year(df, col):
    g = df.groupby("year")[col]
    std = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / std).fillna(0.0)


def _next_season_map(years):
    years = sorted(set(int(y) for y in years))
    return dict(zip(years[:-1], years[1:]))


def build_roster_features(scoring, goalies):
    s = scoring.copy()
    if "lgID" in s.columns:
        s = s[s["lgID"].eq("NHL")]
    for c in ("GP", "Pts"):
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0)

    nxt = _next_season_map(s["year"].unique())

    # ---- player totals per season (all teams) and per (season, team)
    player_season = s.groupby(["playerID", "year"], as_index=False)["Pts"].sum().rename(columns={"Pts": "Pts_all"})
    player_team = s.groupby(["playerID", "year", "tmID"], as_index=False)[["GP", "Pts"]].sum()

    # previous-season points of each player, indexed by the season they apply to
    prev_pts = player_season.assign(year=player_season["year"].map(nxt)).dropna(subset=["year"])
    prev_pts["year"] = prev_pts["year"].astype(int)
    prev_pts = prev_pts.rename(columns={"Pts_all": "prev_Pts"})

    roster = player_team.merge(prev_pts, on=["playerID", "year"], how="left")
    roster["prev_Pts"] = roster["prev_Pts"].fillna(0.0)

    # was the player already on THIS team the previous season?
    prev_team = player_team.assign(year=player_team["year"].map(nxt)).dropna(subset=["year"])
    prev_team["year"] = prev_team["year"].astype(int)
    prev_team = prev_team[["playerID", "year", "tmID", "Pts"]].rename(columns={"Pts": "Pts_with_team_prev"})
    roster = roster.merge(prev_team[["playerID", "year", "tmID"]].assign(was_here=1),
                          on=["playerID", "year", "tmID"], how="left")
    roster["was_here"] = roster["was_here"].fillna(0)

    def team_agg(g):
        top5 = g["prev_Pts"].nlargest(5).sum()
        gp = g["GP"].sum()
        return pd.Series({
            "roster_prev_pts": g["prev_Pts"].sum(),
            "top5_prev_pts": top5,
            "talent_added": g.loc[g["was_here"] == 0, "prev_Pts"].sum(),
            "continuity": (g["GP"] * g["was_here"]).sum() / gp if gp > 0 else np.nan,
        })

    feats = roster.groupby(["year", "tmID"]).apply(team_agg).reset_index()

    # talent lost: points (with that team, previous season) of players who are no longer on the roster
    here_now = roster[["playerID", "year", "tmID"]].assign(still_here=1)
    left = prev_team.merge(here_now, on=["playerID", "year", "tmID"], how="left")
    left = left[left["still_here"].isna()]
    lost = left.groupby(["year", "tmID"], as_index=False)["Pts_with_team_prev"].sum().rename(
        columns={"Pts_with_team_prev": "talent_lost"})
    feats = feats.merge(lost, on=["year", "tmID"], how="left")
    feats["talent_lost"] = feats["talent_lost"].fillna(0.0)

    # ---- goalies: main goalie of the season (most minutes) and his previous-season shrunk save %
    gl = goalies.copy()
    if "lgID" in gl.columns:
        gl = gl[gl["lgID"].eq("NHL")]
    for c in ("Min", "GA", "SA"):
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
    goalie_team = gl.groupby(["playerID", "year", "tmID"], as_index=False)[["Min"]].sum()
    main = (goalie_team.sort_values(["year", "tmID", "Min"], ascending=[True, True, False])
            .drop_duplicates(["year", "tmID"]))
    goalie_season = gl.groupby(["playerID", "year"], as_index=False)[["GA", "SA"]].sum()
    league_prior = gl.groupby("year")[["GA", "SA"]].sum()
    league_prior["lg_sv"] = 1 - league_prior["GA"] / league_prior["SA"].replace(0, np.nan)

    gnxt = _next_season_map(gl["year"].unique())
    prev_g = goalie_season.assign(year=goalie_season["year"].map(gnxt)).dropna(subset=["year"])
    prev_g["year"] = prev_g["year"].astype(int)
    prev_g = prev_g.rename(columns={"GA": "prev_GA", "SA": "prev_SA"})
    prev_lg = league_prior["lg_sv"].rename("prev_lg_sv").to_frame().reset_index()
    prev_lg["year"] = prev_lg["year"].map(gnxt)
    prev_lg = prev_lg.dropna(subset=["year"]).astype({"year": int})

    main = main.merge(prev_g, on=["playerID", "year"], how="left").merge(prev_lg, on="year", how="left")
    shots = main["prev_SA"].fillna(0.0)
    saves = shots - main["prev_GA"].fillna(0.0)
    prior = main["prev_lg_sv"]
    shrunk = (saves + GOALIE_PRIOR_SHOTS * prior) / (shots + GOALIE_PRIOR_SHOTS)
    main["goalie_prev_sv"] = np.where(prior.notna() & (shots > 0), shrunk - prior, 0.0)   # 0 = league-average / unknown

    feats = feats.merge(main[["year", "tmID", "goalie_prev_sv"]], on=["year", "tmID"], how="left")
    feats["goalie_prev_sv"] = feats["goalie_prev_sv"].fillna(0.0)

    # ---- scale-free versions (league-wide z-score per season)
    feats["net_talent"] = feats["talent_added"] - feats["talent_lost"]
    feats["continuity"] = feats["continuity"].fillna(feats["continuity"].mean())
    for raw, z in [("roster_prev_pts", "roster_prev_pts_z"), ("top5_prev_pts", "top5_prev_pts_z"),
                   ("talent_added", "talent_added_z"), ("talent_lost", "talent_lost_z"),
                   ("net_talent", "net_talent_z"), ("continuity", "continuity_z"),
                   ("goalie_prev_sv", "goalie_prev_sv_z")]:
        feats[z] = _z_by_year(feats, raw)
    keep = ["year", "tmID"] + [c for cols in ROSTER_GROUPS.values() for c in cols]
    return feats[keep]
