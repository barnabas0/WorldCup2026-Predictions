"""
World Cup prediction model - data preparation pipeline.

Stage 1: Compute leak-free historical Elo ratings for every team, updated
          match-by-match in chronological order.
Stage 2: Compute leak-free rolling form features (goals for/against, rest days)
          for every team, again strictly using only prior matches.
Stage 3: Split into:
            - train_df: completed matches with known scores, decay-weighted
            - predict_df: the 44 unplayed 2026 World Cup fixtures (NA scores)

Output: two CSVs -> train_features.csv, predict_features.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/results.csv"
OUT_TRAIN = "data/train_features.csv"
OUT_PREDICT = "data/predict_features.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ELO_K = 20                 # base K-factor
ELO_HOME_ADV = 60          # Elo points added to home team's effective rating
ELO_START = 1500
ROLLING_WINDOW = 10        # matches used for rolling goal averages
DECAY_HALF_LIFE_DAYS = 540 # ~18 months, for sample weighting at train time
TRAIN_YEARS_BACK = 6       # how far back to keep rows in train_features.csv
                            # (decay weighting still tapers older matches,
                            #  this is just to bound dataset size / drop
                            #  pre-professional-era noise)

# The dataset's own documentation claims "strictly men's full internationals,"
# but inspection found CONIFA / unrecognized-state competitions (Tamil Eelam,
# Biafra, Mapuche, Ticino, Elba Island, Yoruba Nation, etc. playing in CONIFA
# World Football Cup, Viva World Cup, Tynwald Hill Tournament, World Unity Cup,
# Outrigger Challenge Cup, ConIFA Challenger Cup). These teams never play
# actual FIFA members, so they don't poison anyone's Elo, but they create
# orphan rows with no valid rolling-form history and have no bearing on a
# FIFA World Cup. Dropped entirely by tournament name.
EXCLUDE_TOURNAMENTS = {
    "CONIFA World Football Cup", "CONIFA Asia Cup", "Viva World Cup",
    "CONIFA Africa Football Cup", "Tynwald Hill Tournament", "World Unity Cup",
    "CONIFA South America Football Cup", "Outrigger Challenge Cup",
    "ConIFA Challenger Cup", "CONIFA World Cup qualification",
    "CONIFA World Football Cup qualification",
}


def filter_to_fifa_nations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament-name exclusion lists are fragile - non-FIFA/CONIFA-style teams
    (Raetia, Elba Island, etc.) show up tagged as plain "Friendly" matches
    too, so name-matching the tournament alone misses them.

    Robust approach: build the set of teams that have ever appeared in an
    actual "FIFA World Cup" match (men's, current cycle's name for the
    competition in this dataset) at any point in history. That is about as
    clean a proof of "this is a real FIFA member nation" as exists. Drop any
    match where either team is outside that set.

    This is intentionally strict - it will also drop legitimate FIFA members
    who have simply never qualified for or played in a World Cup match
    (friendlies/qualifiers only). For a World-Cup-specific prediction model
    that's an acceptable trade-off: a nation that's never played a WC match
    isn't a team you need to predict anyway, and excluding them removes a
    long tail of noisy minor-nation friendlies from training.
    """
    wc_mask = df["tournament"].isin(["FIFA World Cup", "FIFA World Cup qualification"])
    fifa_teams = set(df.loc[wc_mask, "home_team"]) | set(df.loc[wc_mask, "away_team"])
    keep_mask = df["home_team"].isin(fifa_teams) & df["away_team"].isin(fifa_teams)
    dropped = len(df) - keep_mask.sum()
    print(f"FIFA-nation filter: kept {keep_mask.sum()} matches, dropped {dropped} "
          f"(non-FIFA-WC-history teams), {len(fifa_teams)} recognized nations")
    return df[keep_mask].copy()


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walks matches in chronological order, updating a running Elo rating per
    team. For each match we RECORD the pre-match Elo (i.e. the rating as it
    stood before this game's result was known) into two new columns,
    home_elo_pre and away_elo_pre. Only AFTER recording do we update the
    ratings using this match's result. This ordering is what prevents
    leakage - a match's features never depend on its own outcome.
    """
    df = df.sort_values("date").reset_index(drop=True)
    elo = {}  # team -> current rating

    home_elo_pre = np.empty(len(df))
    away_elo_pre = np.empty(len(df))

    for i, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        h_elo = elo.get(h, ELO_START)
        a_elo = elo.get(a, ELO_START)

        home_elo_pre[i] = h_elo
        away_elo_pre[i] = a_elo

        # Skip the rating update for unplayed fixtures (NA scores) -
        # they carry no information and must not move anyone's rating.
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        hs, as_ = row["home_score"], row["away_score"]

        # Expected score from home team's perspective, with home advantage
        # folded in as an Elo offset (standard approach in football Elo models)
        diff = (h_elo + ELO_HOME_ADV) - a_elo
        expected_home = 1 / (1 + 10 ** (-diff / 400))

        if hs > as_:
            actual_home = 1.0
        elif hs == as_:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # Goal-difference multiplier (standard World Football Elo extension):
        # bigger margins move ratings more, but with diminishing returns.
        margin = abs(hs - as_)
        if margin <= 1:
            gd_mult = 1.0
        elif margin == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11 + margin) / 8

        k_eff = ELO_K * gd_mult

        elo[h] = h_elo + k_eff * (actual_home - expected_home)
        elo[a] = a_elo + k_eff * ((1 - actual_home) - (1 - expected_home))

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    return df


def compute_rolling_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    For every team, computes rolling average goals-for and goals-against
    over their previous `window` matches, PLUS days since their previous
    match. All values are computed using only matches strictly before the
    current row's date for that team - no leakage.

    Implementation: build a long-format per-team match log (one row per
    team per match, regardless of home/away), sort by team+date, then use
    shift(1).rolling(window) so the current match is never included in its
    own average.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    home_log = df[["match_id", "date", "home_team", "home_score", "away_score"]].copy()
    home_log.columns = ["match_id", "date", "team", "goals_for", "goals_against"]

    away_log = df[["match_id", "date", "away_team", "away_score", "home_score"]].copy()
    away_log.columns = ["match_id", "date", "team", "goals_for", "goals_against"]

    long_log = pd.concat([home_log, away_log], ignore_index=True)
    long_log = long_log.sort_values(["team", "date"]).reset_index(drop=True)

    # Only compute rolling stats over matches with known results - unplayed
    # fixtures must not appear as "prior matches" for later predictions either,
    # but since they're at the end of the dataset chronologically this only
    # matters for the rolling window's own internal correctness.
    played_mask = long_log["goals_for"].notna()

    # goals for/against, shifted so current row excluded, rolling window,
    # computed only over played matches per team
    long_log["roll_gf"] = np.nan
    long_log["roll_ga"] = np.nan
    long_log["days_since_last"] = np.nan

    for team, grp in long_log.groupby("team"):
        idx = grp.index
        played = grp[played_mask.loc[idx]]
        gf_shifted = played["goals_for"].shift(1)
        ga_shifted = played["goals_against"].shift(1)
        roll_gf = gf_shifted.rolling(window, min_periods=1).mean()
        roll_ga = ga_shifted.rolling(window, min_periods=1).mean()
        long_log.loc[played.index, "roll_gf"] = roll_gf
        long_log.loc[played.index, "roll_ga"] = roll_ga

        # For unplayed rows (the prediction fixtures), the relevant rolling
        # stat is just "form going into the match" = full rolling window
        # over ALL prior played matches (no shift needed, since the match
        # itself has no result to leak)
        unplayed = grp[~played_mask.loc[idx]]
        for u_idx, u_row in unplayed.iterrows():
            prior = played[played["date"] < u_row["date"]]
            prior_tail = prior.tail(window)
            if len(prior_tail) > 0:
                long_log.loc[u_idx, "roll_gf"] = prior_tail["goals_for"].mean()
                long_log.loc[u_idx, "roll_ga"] = prior_tail["goals_against"].mean()

        # Days since previous match (any match, played or not, but using
        # only matches strictly before this one)
        all_dates = pd.to_datetime(grp["date"]).reset_index(drop=True)
        days_since = all_dates.diff().dt.days
        long_log.loc[idx, "days_since_last"] = days_since.values

    # Pivot back: join home-side and away-side stats back onto df
    home_stats = long_log[long_log["match_id"].isin(df["match_id"])].copy()

    # Re-split by matching match_id + team back to home/away role
    h_stats = df[["match_id", "home_team"]].merge(
        long_log[["match_id", "team", "roll_gf", "roll_ga", "days_since_last"]],
        left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left"
    )[["match_id", "roll_gf", "roll_ga", "days_since_last"]]
    h_stats.columns = ["match_id", "home_roll_gf", "home_roll_ga", "home_days_since_last"]

    a_stats = df[["match_id", "away_team"]].merge(
        long_log[["match_id", "team", "roll_gf", "roll_ga", "days_since_last"]],
        left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left"
    )[["match_id", "roll_gf", "roll_ga", "days_since_last"]]
    a_stats.columns = ["match_id", "away_roll_gf", "away_roll_ga", "away_days_since_last"]

    df = df.merge(h_stats, on="match_id", how="left")
    df = df.merge(a_stats, on="match_id", how="left")
    df = df.drop(columns=["match_id"])
    return df


def add_decay_weight(df: pd.DataFrame, reference_date) -> pd.DataFrame:
    """
    Sample weight for training: exponential decay based on days between the
    match and a reference date (the most recent date in the dataset / "today").
    This is what implements "recent matches matter most" WITHOUT hard-deleting
    older matches that still carry useful signal (per Dixon-Coles, 1997).
    """
    days_ago = (reference_date - pd.to_datetime(df["date"])).dt.days
    decay_lambda = np.log(2) / DECAY_HALF_LIFE_DAYS
    df["sample_weight"] = np.exp(-decay_lambda * days_ago)
    return df


def main():
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = filter_to_fifa_nations(df)

    print(f"Loaded {len(df)} total matches ({df['date'].min().date()} to {df['date'].max().date()})")

    # --- Elo: computed over FULL history for accuracy, leak-free by construction ---
    df = compute_elo(df)

    # --- Rolling form: also computed over full history so early-2026 matches
    #     have correct rolling stats even though we'll only train on recent years ---
    df = compute_rolling_form(df)

    # --- Derived columns ---
    df["elo_diff"] = (df["home_elo_pre"] + ELO_HOME_ADV) - df["away_elo_pre"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["result"] = np.select(
        [df["home_score"] > df["away_score"],
         df["home_score"] == df["away_score"],
         df["home_score"] < df["away_score"]],
        ["H", "D", "A"],
        default=None
    )
    df["neutral"] = df["neutral"].astype(bool)

    # --- Split: train (played matches, recent years) vs predict (NA scores) ---
    is_unplayed = df["home_score"].isna() | df["away_score"].isna()
    predict_df = df[is_unplayed].copy()

    reference_date = df.loc[~is_unplayed, "date"].max()  # last played match = "today"
    cutoff_date = reference_date - pd.DateOffset(years=TRAIN_YEARS_BACK)

    train_df = df[(~is_unplayed) & (df["date"] >= cutoff_date)].copy()
    train_df = add_decay_weight(train_df, reference_date)

    feature_cols = [
        "date", "home_team", "away_team", "tournament", "neutral",
        "home_elo_pre", "away_elo_pre", "elo_diff",
        "home_roll_gf", "home_roll_ga", "away_roll_gf", "away_roll_ga",
        "home_days_since_last", "away_days_since_last",
    ]

    train_out = train_df[feature_cols + ["home_score", "away_score", "total_goals", "result", "sample_weight"]]
    predict_out = predict_df[feature_cols]

    train_out.to_csv(OUT_TRAIN, index=False)
    predict_out.to_csv(OUT_PREDICT, index=False)

    print(f"\nTrain set: {len(train_out)} matches, {cutoff_date.date()} to {reference_date.date()}")
    print(f"Predict set: {len(predict_out)} unplayed fixtures")
    print(f"\nMissing rolling-form coverage in train set:")
    print(train_out[["home_roll_gf", "away_roll_gf"]].isna().sum())
    print(f"\nSample of predict set:")
    print(predict_out[["date", "home_team", "away_team", "home_elo_pre", "away_elo_pre"]].head(10))


if __name__ == "__main__":
    main()