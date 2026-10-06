"""Feature engineering (paper-inspired, leakage-free).

Every team-performance feature for game g is computed ONLY from games played strictly
before g (groupby(team) -> shift(1) before any rolling/expanding statistic).
The current game's scores are used only to build the target label.
"""
import numpy as np
import pandas as pd
from arenas import ARENAS

RAW = "data/nba_2008-2026.csv"
STATS = ["pts_for", "pts_against", "total_pts", "margin", "win"]


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def load_games(path=RAW):
    d = pd.read_csv(path, parse_dates=["date"])
    d["gid"] = np.arange(len(d))
    d["actual_total"] = d.score_home + d.score_away
    return d.sort_values(["date", "gid"]).reset_index(drop=True)


def team_long(d):
    """One row per (game, team) with that team's own result in that game."""
    h = d[["gid", "date", "season", "home", "away", "score_home", "score_away"]].copy()
    h.columns = ["gid", "date", "season", "team", "opp", "pts_for", "pts_against"]
    h["is_home"] = 1
    a = d[["gid", "date", "season", "away", "home", "score_away", "score_home"]].copy()
    a.columns = ["gid", "date", "season", "team", "opp", "pts_for", "pts_against"]
    a["is_home"] = 0
    t = pd.concat([h, a]).sort_values(["date", "gid"]).reset_index(drop=True)
    t["venue"] = np.where(t.is_home == 1, t.team, t.opp)  # city where the game was played
    t["total_pts"] = t.pts_for + t.pts_against
    t["margin"] = t.pts_for - t.pts_against
    t["win"] = (t.margin > 0).astype(int)
    return t


def add_team_history(t):
    g = t.groupby("team", sort=False)
    # season-to-date averages BEFORE the game (expanding mean, shifted by one game)
    gs = t.groupby(["team", "season"], sort=False)
    for c in ["pts_for", "pts_against", "total_pts"]:
        t[f"sd_{c}"] = gs[c].transform(lambda s: s.shift(1).expanding().mean())
    # rest / schedule density
    t["prev_date"] = g["date"].shift(1)
    t["rest_days"] = (t["date"] - t["prev_date"]).dt.days.clip(upper=10)
    t["b2b"] = (t["rest_days"] == 1).astype(float).where(t["rest_days"].notna())

    def games_last7(x):
        s = pd.Series(1, index=x.values)
        return pd.Series(s.rolling("7D", closed="left").sum().values, index=x.index)

    t["games_last7"] = g["date"].transform(games_last7).fillna(0)
    # travel: km between previous game venue and this game venue
    t["prev_venue"] = g["venue"].shift(1)
    t["travel_km"] = [haversine_km(ARENAS[p], ARENAS[v]) if isinstance(p, str) else np.nan
                      for p, v in zip(t.prev_venue, t.venue)]
    # opponent strength: opponent's season-to-date averages before THIS game
    opp = t[["gid", "team", "sd_pts_for", "sd_pts_against"]].rename(
        columns={"team": "opp", "sd_pts_for": "opp_sd_for", "sd_pts_against": "opp_sd_against"})
    t = t.merge(opp, on=["gid", "opp"], how="left").sort_values(["date", "gid"]).reset_index(drop=True)
    g = t.groupby("team", sort=False)
    # previous-3-game rolling means (shift(1) => strictly earlier games)
    for c in STATS + ["opp_sd_for", "opp_sd_against"]:
        t[f"l3_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    return t


FEATS_TEAM = (["l3_" + c for c in STATS + ["opp_sd_for", "opp_sd_against"]]
              + ["sd_pts_for", "sd_pts_against", "sd_total_pts",
                 "rest_days", "b2b", "games_last7", "travel_km"])


def build_dataset(path=RAW):
    d = load_games(path)
    t = add_team_history(team_long(d))
    keep = ["gid"] + FEATS_TEAM
    H = t[t.is_home == 1][keep].set_index("gid").add_prefix("h_").reset_index()
    A = t[t.is_home == 0][keep].set_index("gid").add_prefix("a_").reset_index()
    X = d.merge(H, on="gid").merge(A, on="gid")
    # combined / market features
    X["line_total"] = X["total"]  # sportsbook Over/Under line
    X["abs_spread"] = X["spread"].abs()
    X["home_favored"] = (X["whos_favored"] == "home").astype(float)
    X["is_playoff"] = X["playoffs"].astype(int)
    X["exp_total_l3"] = (X.h_l3_pts_for + X.a_l3_pts_against + X.a_l3_pts_for + X.h_l3_pts_against) / 2
    X["exp_total_sd"] = (X.h_sd_pts_for + X.a_sd_pts_against + X.a_sd_pts_for + X.h_sd_pts_against) / 2
    X["l3_total_minus_line"] = X.exp_total_l3 - X.line_total
    X["sd_total_minus_line"] = X.exp_total_sd - X.line_total
    X["both_b2b"] = X.h_b2b * X.a_b2b
    X["travel_diff_km"] = X.a_travel_km - X.h_travel_km
    # target (the current game's score is used ONLY here)
    X = X[X.actual_total != X.total].copy()  # drop pushes
    X["over"] = (X.actual_total > X.total).astype(int)  # 1 = OVER, 0 = UNDER
    return X.sort_values(["date", "gid"]).reset_index(drop=True)


FEATURES = (["h_" + c for c in FEATS_TEAM] + ["a_" + c for c in FEATS_TEAM]
            + ["line_total", "abs_spread", "home_favored", "is_playoff",
               "exp_total_l3", "exp_total_sd", "l3_total_minus_line", "sd_total_minus_line",
               "both_b2b", "travel_diff_km"])


def split(X):
    tr = X[X.season <= 2022]
    va = X[X.season == 2023]
    te = X[X.season >= 2024]
    # a team's first 3 games ever have no last-3 stats (only the very first games of 2008)
    ok = lambda df: df[df.h_l3_pts_for.notna() & df.a_l3_pts_for.notna()]
    return ok(tr), ok(va), ok(te)
