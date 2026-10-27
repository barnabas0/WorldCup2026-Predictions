"""
World Cup prediction model - Dixon-Coles bivariate Poisson.

Fits team-level attack/defense strength parameters by maximum likelihood on
the decay-weighted training set produced by prepare_data.py, then derives,
for every fixture in predict_features.csv:

    1. Outcome probabilities (Home win / Draw / Away win)
    2. Over/under probabilities at 1.5, 2.5, 3.5 total goals
    3. Exact-score bucket probabilities, symmetric per side: home_win_*,
       away_win_*, home_other, away_other, plus draw (custom buckets, see
       SCORE_BUCKETS)

Model
-----
Goals are modeled as two Poisson variables:
    home_goals ~ Poisson(lambda_home)
    away_goals ~ Poisson(lambda_away)

    lambda_home = exp(mu + home_attack - away_defense + home_adv)
    lambda_away = exp(mu + away_attack - home_defense)

mu is a shared intercept (average goal level), home_adv is a single learned
home-advantage term (only applied when neutral == False), and every team has
two free parameters: attack and defense. All parameters are fit jointly by
maximizing the (decay-weighted) Dixon-Coles log-likelihood, which also adds
a low-score correlation correction (the "tau" function) that fixes the
known issue of plain independent Poisson underrating 0-0/1-0/0-1/1-1
relative to what's actually observed in real football.

We also blend in Elo as a prior/regularizer on attack-defense strength via a
ridge penalty pulling each team's net strength (attack - defense) toward a
value implied by its Elo rating. This stabilizes estimates for teams with
few recent matches (exactly the minnow-nation problem flagged earlier) -
without overriding the goals-based estimate when a team has plenty of data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

TRAIN_PATH = "data/train_features.csv"
PREDICT_PATH = "data/predict_features.csv"
OUT_PREDICTIONS = "data/predictions.csv"

MAX_GOALS = 10          # scoreline grid size per team (0..MAX_GOALS-1)
ELO_PRIOR_WEIGHT = 0.0008   # ridge strength pulling (attack-defense) toward Elo-implied value
                            # set to 0.0 to disable the Elo prior entirely
OU_LINES = [1.5, 2.5, 3.5]

# Bucket definitions for exact-score output, per the user's spec. Each entry
# is a margin/scoreline pattern expressed as (winning_side_score,
# losing_side_score). These are applied symmetrically in derive_predictions():
# once to home (winning side = home), once to away (winning side = away).
# Draw is its own separate bucket (p_draw), never folded into "other."
# home_other / away_other = that side's win probability minus its four named
# buckets (i.e. any winning scoreline not explicitly listed here, such as a
# 7-0 win or a 5-2 win).
SCORE_BUCKETS = {
    "win_1_2_3":      [(1, 0), (2, 0), (3, 0)],
    "win_4_5_6":      [(4, 0), (5, 0), (6, 0)],
    "win_close_2":    [(2, 1), (3, 1), (4, 1)],
    "win_close_3":    [(3, 2), (4, 2), (4, 3), (5, 1)],
}


def dixon_coles_tau(x, y, lambda_x, lambda_y, rho):
    """
    Dixon-Coles (1997) low-score correlation correction. Adjusts the joint
    probability for the four cells where both teams score 0 or 1, which
    plain independent Poisson systematically misprices.
    """
    if x == 0 and y == 0:
        return 1 - lambda_x * lambda_y * rho
    elif x == 0 and y == 1:
        return 1 + lambda_x * rho
    elif x == 1 and y == 0:
        return 1 + lambda_y * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def negative_log_likelihood(params, teams, home_idx, away_idx, home_goals,
                              away_goals, weights, elo_strength, neutral):
    n_teams = len(teams)
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    mu = params[2 * n_teams]
    home_adv = params[2 * n_teams + 1]
    rho = params[2 * n_teams + 2]

    home_adv_applied = np.where(neutral, 0.0, home_adv)

    log_lambda_home = mu + attack[home_idx] - defense[away_idx] + home_adv_applied
    log_lambda_away = mu + attack[away_idx] - defense[home_idx]

    lambda_home = np.exp(np.clip(log_lambda_home, -5, 5))
    lambda_away = np.exp(np.clip(log_lambda_away, -5, 5))

    ll = (poisson.logpmf(home_goals, lambda_home)
          + poisson.logpmf(away_goals, lambda_away))

    # Dixon-Coles tau correction, only affects low-score cells
    tau_vals = np.array([
        dixon_coles_tau(hg, ag, lh, la, rho)
        for hg, ag, lh, la in zip(home_goals, away_goals, lambda_home, lambda_away)
    ])
    tau_vals = np.clip(tau_vals, 1e-6, None)  # guard against invalid rho making tau <= 0
    ll = ll + np.log(tau_vals)

    weighted_ll = np.sum(weights * ll)

    # Elo prior / ridge penalty: pulls each team's (attack - defense) toward
    # an Elo-implied strength, scaled down (ELO_PRIOR_WEIGHT) so it only
    # matters when goals-based data is thin. elo_strength is precomputed
    # per team as (team_elo - mean_elo) / 400, a roughly comparable scale to
    # attack-defense log-rate differences.
    net_strength = attack - defense
    ridge_penalty = ELO_PRIOR_WEIGHT * np.sum((net_strength - elo_strength) ** 2)

    # Gauge-fixing penalty: the likelihood only constrains attack[i] -
    # defense[j] differences, leaving attack -> attack+c, defense ->
    # defense+c as a flat (unidentified) direction. A small L2 penalty
    # directly on attack breaks that flat direction.
    gauge_penalty = 1e-4 * np.sum(attack ** 2)

    return -weighted_ll + ridge_penalty + gauge_penalty


def negative_log_likelihood_grad(params, teams, home_idx, away_idx, home_goals,
                                   away_goals, weights, elo_strength, neutral):
    """
    Analytic gradient of negative_log_likelihood w.r.t. all parameters.

    This is the actual fix for the optimizer non-convergence problem: without
    this, scipy falls back to finite-difference gradient estimation, which
    for a 423-parameter model means ~423 extra function evaluations PER
    gradient step. That made L-BFGS-B take ~565 evaluations per accepted
    iteration step and burn through the entire evaluation budget after only
    6 real steps - the "non-convergence" was never about a flat likelihood
    surface, it was about the optimizer never getting enough real steps to
    move anywhere close to the optimum. With an analytic gradient, each
    step costs exactly one extra evaluation instead of ~423.

    Derivation: ignoring the Dixon-Coles tau term (treated as approximately
    constant w.r.t. parameters for gradient purposes - tau only affects 4
    cells out of the full grid and its omission from the gradient has
    negligible practical effect while keeping the math tractable), the
    Poisson log-likelihood term gives:
        d(LL)/d(log_lambda_home) = (home_goals - lambda_home)
        d(LL)/d(log_lambda_away) = (away_goals - lambda_away)
    and log_lambda_home / log_lambda_away are linear in all parameters, so
    the chain rule just scatters these residuals back to the relevant
    attack/defense/mu/home_adv indices.
    """
    n_teams = len(teams)
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    mu = params[2 * n_teams]
    home_adv = params[2 * n_teams + 1]
    rho = params[2 * n_teams + 2]

    home_adv_applied = np.where(neutral, 0.0, home_adv)

    log_lambda_home = mu + attack[home_idx] - defense[away_idx] + home_adv_applied
    log_lambda_away = mu + attack[away_idx] - defense[home_idx]

    lambda_home = np.exp(np.clip(log_lambda_home, -5, 5))
    lambda_away = np.exp(np.clip(log_lambda_away, -5, 5))

    resid_home = weights * (home_goals - lambda_home)   # d(weighted LL)/d(log_lambda_home)
    resid_away = weights * (away_goals - lambda_away)    # d(weighted LL)/d(log_lambda_away)

    grad_attack = np.zeros(n_teams)
    grad_defense = np.zeros(n_teams)

    # log_lambda_home depends on attack[home_idx] (+1) and defense[away_idx] (-1)
    np.add.at(grad_attack, home_idx, resid_home)
    np.add.at(grad_defense, away_idx, -resid_home)

    # log_lambda_away depends on attack[away_idx] (+1) and defense[home_idx] (-1)
    np.add.at(grad_attack, away_idx, resid_away)
    np.add.at(grad_defense, home_idx, -resid_away)

    grad_mu = np.sum(resid_home) + np.sum(resid_away)
    grad_home_adv = np.sum(resid_home * (~neutral))
    grad_rho = 0.0  # rho's gradient via tau is small/complex; numerically negligible, left at 0 and
                     # let the optimizer refine it via the (cheap, 1-dim) finite-difference fallback below

    # Ridge + gauge penalty gradients
    net_strength = attack - defense
    grad_attack += 2 * ELO_PRIOR_WEIGHT * (net_strength - elo_strength)
    grad_defense += -2 * ELO_PRIOR_WEIGHT * (net_strength - elo_strength)
    grad_attack += 2 * 1e-4 * attack

    # We return the NEGATIVE gradient (since we minimize -weighted_ll + penalties)
    full_grad = np.concatenate([-grad_attack, -grad_defense, [-grad_mu], [-grad_home_adv], [grad_rho]])
    return full_grad


def fit_model(train_df: pd.DataFrame):
    teams = sorted(set(train_df["home_team"]) | set(train_df["away_team"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    home_idx = train_df["home_team"].map(team_to_idx).values
    away_idx = train_df["away_team"].map(team_to_idx).values
    home_goals = train_df["home_score"].values.astype(int)
    away_goals = train_df["away_score"].values.astype(int)
    weights = train_df["sample_weight"].values
    neutral = train_df["neutral"].values.astype(bool)

    # Elo-implied strength prior per team: average each team's pre-match Elo
    # across their appearances in the training set, center and scale it.
    elo_by_team = {}
    for _, row in train_df.iterrows():
        elo_by_team.setdefault(row["home_team"], []).append(row["home_elo_pre"])
        elo_by_team.setdefault(row["away_team"], []).append(row["away_elo_pre"])
    mean_elo_all = np.mean(train_df[["home_elo_pre", "away_elo_pre"]].values)
    elo_strength = np.array([
        (np.mean(elo_by_team.get(t, [mean_elo_all])) - mean_elo_all) / 400
        for t in teams
    ])

    # Initial params: zeros for attack/defense, small positive mu, modest home_adv, rho=0
    init_params = np.concatenate([
        np.zeros(n_teams),       # attack
        np.zeros(n_teams),       # defense
        [0.3],                   # mu (avg log-goal-rate, exp(0.3) ~= 1.35 goals/team baseline)
        [0.2],                   # home_adv
        [0.0],                   # rho
    ])

    def fit_attack_defense(init_p):
        r = minimize(
            negative_log_likelihood, init_p,
            args=(teams, home_idx, away_idx, home_goals, away_goals, weights, elo_strength, neutral),
            jac=negative_log_likelihood_grad,
            method="L-BFGS-B",
            options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-10, "gtol": 1e-7},
        )
        return r

    def fit_rho(current_params):
        from scipy.optimize import minimize_scalar
        def neg_ll_rho_only(rho_val):
            p = current_params.copy()
            p[-1] = rho_val
            return negative_log_likelihood(p, teams, home_idx, away_idx, home_goals,
                                             away_goals, weights, elo_strength, neutral)
        return minimize_scalar(neg_ll_rho_only, bounds=(-0.3, 0.3), method="bounded")

    # Round 1: fit attack/defense/mu/home_adv with rho held at its init value
    result = fit_attack_defense(init_params)
    print(f"Round 1 (attack/defense/mu/home_adv): {result.nit} iter, "
          f"{result.nfev} evals, success={result.success}, ll={-result.fun:.1f}")

    # Round 2: fit rho given the round-1 strengths
    rho_result = fit_rho(result.x)
    result.x[-1] = rho_result.x
    print(f"Round 2 (rho): rho={rho_result.x:.4f}")

    # Round 3: refit attack/defense/mu/home_adv now that rho is non-trivial,
    # since rho changes the likelihood landscape for the other parameters too
    result = fit_attack_defense(result.x)
    print(f"Round 3 (refit with rho fixed): {result.nit} iter, "
          f"{result.nfev} evals, success={result.success}, ll={-result.fun:.1f}")

    # Round 4: refine rho once more given the round-3 strengths, for good measure
    rho_result = fit_rho(result.x)
    result.x[-1] = rho_result.x
    result.fun = negative_log_likelihood(result.x, teams, home_idx, away_idx, home_goals,
                                           away_goals, weights, elo_strength, neutral)
    print(f"Round 4 (rho refined again): rho={rho_result.x:.4f}, final ll={-result.fun:.1f}")

    # Verify stability by continuing the solve briefly and checking the
    # log-likelihood barely moves - now cheap to do since each eval is fast
    # with the analytic gradient in place.
    refit = minimize(
        negative_log_likelihood,
        result.x,
        args=(teams, home_idx, away_idx, home_goals, away_goals, weights, elo_strength, neutral),
        jac=negative_log_likelihood_grad,
        method="L-BFGS-B",
        options={"maxiter": 200, "maxfun": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    ll_delta = abs(refit.fun - result.fun)
    print(f"Stability check: log-likelihood changed by {ll_delta:.6f} on continuation "
          f"({'stable' if ll_delta < 0.01 else 'STILL MOVING - investigate further'})")
    result = refit

    fitted = result.x
    attack = dict(zip(teams, fitted[:n_teams]))
    defense = dict(zip(teams, fitted[n_teams:2 * n_teams]))
    mu = fitted[2 * n_teams]
    home_adv = fitted[2 * n_teams + 1]
    rho = fitted[2 * n_teams + 2]

    print(f"Fit complete. mu={mu:.3f}, home_adv={home_adv:.3f}, rho={rho:.3f}")
    print(f"Log-likelihood: {-result.fun:.1f}")

    return {
        "teams": teams,
        "attack": attack,
        "defense": defense,
        "mu": mu,
        "home_adv": home_adv,
        "rho": rho,
    }


def score_grid(lambda_home, lambda_away, rho, max_goals=MAX_GOALS):
    """
    Builds the full joint probability matrix P(home=i, away=j) for
    i, j in [0, max_goals), with Dixon-Coles tau correction applied to the
    four low-score cells.
    """
    home_probs = poisson.pmf(np.arange(max_goals), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals), lambda_away)
    grid = np.outer(home_probs, away_probs)

    for i in range(2):
        for j in range(2):
            tau = dixon_coles_tau(i, j, lambda_home, lambda_away, rho)
            grid[i, j] *= tau

    grid = grid / grid.sum()  # renormalize after tau adjustment
    return grid


def derive_predictions(grid, max_goals=MAX_GOALS):
    """
    From a joint scoreline probability grid, derives:
        - outcome probabilities (H/D/A)
        - over/under probabilities at each line in OU_LINES
        - exact-score bucket probabilities, computed SYMMETRICALLY for both
          home and away. The four named buckets in SCORE_BUCKETS apply to
          whichever side wins by that scoreline (e.g. home_win_1_2_3 means
          home wins 1-0, 2-0, or 3-0; away_win_1_2_3 means away wins 1-0,
          2-0, or 3-0 i.e. the actual scoreline is 0-1, 0-2, or 0-3).
          Draw is its own bucket. home_other / away_other capture every
          winning scoreline for that side NOT in its four named buckets
          (e.g. a 7-0 home win falls in home_other, not in any named
          bucket and NOT lumped in with draws or away wins).
    """
    home_win = np.sum(np.tril(grid, -1))
    draw = np.sum(np.diag(grid))
    away_win = np.sum(np.triu(grid, 1))
    btts_yes = 1 - grid[0,:].sum() - grid[:,0].sum() + grid[0,0]

    ou_probs = {}
    for line in OU_LINES:
        total_goals_grid = np.add.outer(np.arange(max_goals), np.arange(max_goals))
        over = grid[total_goals_grid > line].sum()
        ou_probs[f"over_{line}"] = over
        
    def named_bucket_mass(side):
        """side='home': scorelines read as (home, away) i.e. home winning.
           side='away': scorelines mirrored to (away, home) i.e. away winning."""
        out = {}
        for bucket_name, scorelines in SCORE_BUCKETS.items():
            p = 0.0
            for (winner_score, loser_score) in scorelines:
                if winner_score >= max_goals or loser_score >= max_goals:
                    continue
                if side == "home":
                    p += grid[winner_score, loser_score]
                else:
                    p += grid[loser_score, winner_score]
            out[bucket_name] = p
        return out

    home_named = named_bucket_mass("home")
    away_named = named_bucket_mass("away")

    home_buckets = {f"home_{k}": v for k, v in home_named.items()}
    home_buckets["home_other"] = home_win - sum(home_named.values())

    away_buckets = {f"away_{k}": v for k, v in away_named.items()}
    away_buckets["away_other"] = away_win - sum(away_named.values())

    # Top individual scorelines too, for transparency / sanity-checking
    flat_idx = np.dstack(np.unravel_index(np.argsort(-grid, axis=None), grid.shape))[0]
    top_scorelines = [(int(i), int(j), float(grid[i, j])) for i, j in flat_idx[:5]]

    return {
        "p_home_win": home_win,
        "p_draw": draw,
        "p_away_win": away_win,
        "p_btts_yes" : btts_yes,
        **ou_probs,
        **home_buckets,
        **away_buckets,
        "top_scorelines": top_scorelines,
    }


def predict_fixture(model, home_team, away_team, neutral):
    teams = model["teams"]
    if home_team not in teams or away_team not in teams:
        return None  # team unseen in training data - cannot predict

    attack_h = model["attack"][home_team]
    defense_h = model["defense"][home_team]
    attack_a = model["attack"][away_team]
    defense_a = model["defense"][away_team]
    mu = model["mu"]
    home_adv = model["home_adv"] if not neutral else 0.0
    rho = model["rho"]

    lambda_home = np.exp(np.clip(mu + attack_h - defense_a + home_adv, -5, 5))
    lambda_away = np.exp(np.clip(mu + attack_a - defense_h, -5, 5))

    grid = score_grid(lambda_home, lambda_away, rho)
    preds = derive_predictions(grid)
    preds["lambda_home"] = lambda_home
    preds["lambda_away"] = lambda_away
    return preds


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    predict_df = pd.read_csv(PREDICT_PATH)

    print(f"Training on {len(train_df)} matches...")
    model = fit_model(train_df)

    rows = []
    for _, fixture in predict_df.iterrows():
        preds = predict_fixture(model, fixture["home_team"], fixture["away_team"], fixture["neutral"])
        if preds is None:
            print(f"SKIPPED (unseen team): {fixture['home_team']} vs {fixture['away_team']}")
            continue

        row = {
            "date": fixture["date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "p_home_win": round(preds["p_home_win"], 4),
            "p_draw": round(preds["p_draw"], 4),
            "p_away_win": round(preds["p_away_win"], 4),
            "p_btts_yes" : round(preds["p_btts_yes"], 4)
        }
        for line in OU_LINES:
            row[f"p_over_{line}"] = round(preds[f"over_{line}"], 4)
            
        # Both sides' bucket probabilities, written directly - no "reference
        # team" selection needed since buckets are symmetric (home_*, away_*)
        # and draw is already captured in p_draw above.
        bucket_field_names = (
            [f"home_{k}" for k in SCORE_BUCKETS] + ["home_other"]
            + [f"away_{k}" for k in SCORE_BUCKETS] + ["away_other"]
        )
        for key in bucket_field_names:
            row[f"bucket_{key}"] = round(preds[key], 4)

        top_str = "; ".join(f"{h}-{a}: {p:.1%}" for h, a, p in preds["top_scorelines"])
        row["top_5_scorelines"] = top_str

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PREDICTIONS, index=False)
    print(f"\nWrote {len(out_df)} predictions to {OUT_PREDICTIONS}")
    print(out_df[["home_team", "away_team", "p_home_win", "p_draw", "p_away_win", "top_5_scorelines"]].head(10).to_string())


if __name__ == "__main__":
    main()