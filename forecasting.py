"""
SCIIP - Forecasting Engine
Produces a monthly demand-rate forecast per (project, part_code), using a
different method depending on the part's Syntetos-Boylan classification.

    Smooth / Erratic    -> Simple Exponential Smoothing (SES)
    Intermittent / Lumpy -> TSB (Teunter-Syntetos-Babai)
                            separately smooths (a) the PROBABILITY demand
                            occurs in a given period (updated every period,
                            including quiet ones - so it correctly decays
                            for parts that have gone quiet), and (b) the
                            SIZE of demand when it does occur. Forecast =
                            probability x size.

All methods forecast a constant monthly RATE going forward (no trend/
seasonality modeling yet).
"""
import numpy as np
import pandas as pd

ALPHA_CROSTON = 0.1
ALPHA_SES = 0.2


def _simple_exp_smoothing(values: np.ndarray, alpha: float = ALPHA_SES) -> float:
    """Returns the smoothed level (= forecast for next period)."""
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return level


def _tsb(values: np.ndarray, alpha: float = 0.1, beta: float = 0.1):
    """
    Teunter-Syntetos-Babai method - the literal two-stage model:
    (1) a smoothed PROBABILITY that demand occurs in a given period, updated
        every single period.
    (2) a smoothed demand SIZE, updated only in periods demand occurs.
    Forecast = probability x size.
    Returns (occurrence_probability, expected_size_if_occurs, forecast_rate, method_name)
    """
    if len(values) == 0 or np.all(values == 0):
        return 0.0, 0.0, 0.0, "No demand history"

    first_idx = np.nonzero(values)[0][0]
    p = 1.0 / (first_idx + 1)
    z = values[first_idx]

    for t in range(first_idx + 1, len(values)):
        occurred = values[t] > 0
        p = p + beta * ((1.0 if occurred else 0.0) - p)
        if occurred:
            z = z + alpha * (values[t] - z)

    rate = p * z
    return round(p, 3), round(z, 2), rate, "TSB (occurrence probability x size)"


def generate_forecasts(monthly_demand: pd.DataFrame, pattern_classification: pd.DataFrame,
                        tuned_params: pd.DataFrame = None, horizon_months: int = 3) -> pd.DataFrame:
    """
    tuned_params: optional output of tune_forecast_parameters() - when given,
        each part uses ITS OWN tuned alpha/beta/threshold instead of the
        fixed defaults.
    """
    pattern_lookup = pattern_classification.set_index(["project", "part_code"])["demand_pattern"].to_dict()
    params_lookup = {}
    if tuned_params is not None and len(tuned_params) > 0:
        for _, row in tuned_params.iterrows():
            params_lookup[(row["project"], row["part_code"])] = (row["tuned_alpha"], row["tuned_beta"], row["tuned_threshold"])

    records = []
    for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
        g = g.sort_values("month")
        values = g["qty"].to_numpy(dtype=float)
        pattern = pattern_lookup.get((project, part_code), "Unknown")
        occurrence_prob, expected_size, predicted_to_occur = None, None, None
        tuned_alpha, tuned_beta, tuned_threshold = params_lookup.get((project, part_code), (None, None, None))

        if pattern in ("Smooth", "Erratic"):
            alpha = tuned_alpha if tuned_alpha is not None else ALPHA_SES
            rate = _simple_exp_smoothing(values, alpha=alpha)
            method = "Simple Exponential Smoothing" + (" (tuned)" if tuned_alpha is not None else " (default)")
            predicted_to_occur = True
        elif pattern in ("Intermittent", "Lumpy"):
            alpha = tuned_alpha if tuned_alpha is not None else ALPHA_CROSTON
            beta = tuned_beta if tuned_beta is not None else ALPHA_CROSTON
            threshold = tuned_threshold if tuned_threshold is not None else 0.5
            occurrence_prob, expected_size, rate, method = _tsb(values, alpha=alpha, beta=beta)
            method += " (tuned)" if tuned_alpha is not None else " (default)"
            predicted_to_occur = occurrence_prob >= threshold
        else:
            rate, method = 0.0, "Unclassified"

        rate = max(0.0, rate)
        records.append({
            "project": project,
            "part_code": part_code,
            "demand_pattern": pattern,
            "forecast_method": method,
            "months_of_history": len(values),
            "occurrence_probability_this_month": occurrence_prob,
            "predicted_to_occur_this_month": predicted_to_occur,
            "expected_qty_if_it_occurs": expected_size,
            "forecast_monthly_qty": round(rate, 2),
            f"forecast_next_{horizon_months}mo_qty": round(rate * horizon_months, 2),
        })

    return pd.DataFrame(records).sort_values(f"forecast_next_{horizon_months}mo_qty", ascending=False)


def generate_forecast_horizon(monthly_demand: pd.DataFrame, pattern_classification: pd.DataFrame,
                                tuned_params: pd.DataFrame = None, num_months: int = 6) -> pd.DataFrame:
    """
    Unrolls the forecast into one row per (project, part_code, future calendar
    month) instead of a single aggregate total - answers "which of the next
    N months is demand expected in, and how much each time" directly.

    Honest note: TSB/SES forecast a STATIONARY rate - without enough history
    to detect real seasonality, the model's best estimate is genuinely the
    same probability/size for month+1 through month+N. This function doesn't
    invent variation that isn't there; it makes the flat rate legible against
    real calendar months instead of hiding it in one aggregate number.
    """
    pattern_lookup = pattern_classification.set_index(["project", "part_code"])["demand_pattern"].to_dict()
    params_lookup = {}
    if tuned_params is not None and len(tuned_params) > 0:
        for _, row in tuned_params.iterrows():
            params_lookup[(row["project"], row["part_code"])] = (row["tuned_alpha"], row["tuned_beta"], row["tuned_threshold"])

    records = []
    for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
        g = g.sort_values("month")
        values = g["qty"].to_numpy(dtype=float)
        last_month = g["month"].max()
        pattern = pattern_lookup.get((project, part_code), "Unknown")
        tuned_alpha, tuned_beta, tuned_threshold = params_lookup.get((project, part_code), (None, None, None))

        occurrence_prob, predicted_to_occur = None, None
        if pattern in ("Smooth", "Erratic"):
            alpha = tuned_alpha if tuned_alpha is not None else ALPHA_SES
            rate = _simple_exp_smoothing(values, alpha=alpha)
            predicted_to_occur = True
        elif pattern in ("Intermittent", "Lumpy"):
            alpha = tuned_alpha if tuned_alpha is not None else ALPHA_CROSTON
            beta = tuned_beta if tuned_beta is not None else ALPHA_CROSTON
            threshold = tuned_threshold if tuned_threshold is not None else 0.5
            occurrence_prob, expected_size, rate, _ = _tsb(values, alpha=alpha, beta=beta)
            predicted_to_occur = occurrence_prob >= threshold
        else:
            rate = 0.0

        rate = max(0.0, rate)
        cumulative = 0.0
        for i in range(1, num_months + 1):
            future_month = last_month + i
            cumulative += rate
            records.append({
                "project": project,
                "part_code": part_code,
                "months_ahead": i,
                "forecast_month": str(future_month),
                "predicted_to_occur": predicted_to_occur,
                "occurrence_probability": occurrence_prob,
                "forecast_qty": round(rate, 2),
                "cumulative_qty": round(cumulative, 2),
            })

    return pd.DataFrame(records)


def build_part_forecast_pivot(horizon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide reference table: one row per (project, part_code), one column per
    future calendar month (chronological), values = forecasted qty, plus a
    Total column - the forward-looking twin of build_part_monthly_pivot().
    """
    if len(horizon_df) == 0:
        return pd.DataFrame()
    month_order = horizon_df.drop_duplicates("forecast_month").sort_values("months_ahead")["forecast_month"].tolist()
    pivot = horizon_df.pivot_table(index=["project", "part_code"], columns="forecast_month", values="forecast_qty")
    pivot = pivot[month_order].reset_index()
    pivot["Total_forecast"] = pivot[month_order].sum(axis=1)
    return pivot.sort_values("Total_forecast", ascending=False)


def build_part_occurrence_pivot(horizon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide reference table: one row per (project, part_code), one column per
    future calendar month, values = Yes/No/N-A - directly answers "out of
    the next N months, which ones is demand actually expected in."
    """
    if len(horizon_df) == 0:
        return pd.DataFrame()
    df = horizon_df.copy()
    df["occurs_label"] = df["predicted_to_occur"].map({True: "Yes", False: "No"})
    df["occurs_label"] = df["occurs_label"].fillna("N/A")  # only unclassified parts (no pattern) fall through to this
    month_order = df.drop_duplicates("forecast_month").sort_values("months_ahead")["forecast_month"].tolist()
    pivot = df.pivot_table(index=["project", "part_code"], columns="forecast_month", values="occurs_label", aggfunc="first")
    pivot = pivot[month_order].reset_index()
    return pivot


def backtest_accuracy(monthly_demand: pd.DataFrame, pattern_classification: pd.DataFrame,
                        min_history: int = 6) -> pd.DataFrame:
    """
    Walk-forward validation: at each historical point with enough prior
    months, forecast the NEXT month using only data available up to that
    point, then compare against what actually happened.
    """
    pattern_lookup = pattern_classification.set_index(["project", "part_code"])["demand_pattern"].to_dict()
    records = []

    for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
        g = g.sort_values("month")
        values = g["qty"].to_numpy(dtype=float)
        pattern = pattern_lookup.get((project, part_code), "Unknown")
        if len(values) < min_history + 1:
            continue

        hits, magnitude_errors = [], []
        for t in range(min_history, len(values) - 1):
            history = values[:t + 1]
            actual_next = values[t + 1]
            actual_occurred = actual_next > 0

            if pattern in ("Smooth", "Erratic"):
                forecast_qty = max(0.0, _simple_exp_smoothing(history))
                predicted_occurrence = True
            elif pattern in ("Intermittent", "Lumpy"):
                prob, size, forecast_qty, _ = _tsb(history)
                predicted_occurrence = prob >= 0.5
            else:
                continue

            hits.append(predicted_occurrence == actual_occurred)
            if actual_occurred:
                pct_err = abs(forecast_qty - actual_next) / actual_next * 100
                magnitude_errors.append(pct_err)

        if not hits:
            continue

        records.append({
            "project": project,
            "part_code": part_code,
            "demand_pattern": pattern,
            "num_test_points": len(hits),
            "occurrence_hit_rate_pct": round(100 * sum(hits) / len(hits), 1),
            "avg_magnitude_error_pct": round(sum(magnitude_errors) / len(magnitude_errors), 1) if magnitude_errors else None,
        })

    if not records:
        return pd.DataFrame(columns=["project", "part_code", "demand_pattern", "num_test_points",
                                       "occurrence_hit_rate_pct", "avg_magnitude_error_pct"])
    return pd.DataFrame(records).sort_values("num_test_points", ascending=False)


def build_part_monthly_pivot(monthly_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Reference table: one row per (project, part_code), one column per
    calendar month in chronological order, values = qty ordered that
    month, plus a Total column at the end.
    """
    df = monthly_demand.copy()
    df["month"] = df["month"].astype(str)
    pivot = df.pivot_table(index=["project", "part_code"], columns="month", values="qty", fill_value=0)
    month_cols = sorted(pivot.columns)
    pivot = pivot[month_cols].reset_index()
    pivot["Total"] = pivot[month_cols].sum(axis=1)
    return pivot.sort_values("Total", ascending=False)


def _naive_baseline_hit_rate(values: np.ndarray, min_history: int) -> float:
    """
    The score a completely naive strategy gets: always predict whichever
    outcome (demand / no demand) was more common in the TRAINING history at
    each point.
    """
    hits = []
    for t in range(min_history, len(values) - 1):
        history = values[:t + 1]
        occurrence_rate = np.mean(history > 0)
        naive_predicts_occurrence = occurrence_rate >= 0.5
        actual_occurred = values[t + 1] > 0
        hits.append(naive_predicts_occurrence == actual_occurred)
    return round(100 * sum(hits) / len(hits), 1) if hits else None


def _backtest_one_config(values: np.ndarray, pattern: str, min_history: int,
                           alpha: float, beta: float = None, threshold: float = 0.5):
    hits, magnitude_errors = [], []
    for t in range(min_history, len(values) - 1):
        history = values[:t + 1]
        actual_next = values[t + 1]
        actual_occurred = actual_next > 0

        if pattern in ("Smooth", "Erratic"):
            forecast_qty = max(0.0, _simple_exp_smoothing(history, alpha=alpha))
            predicted_occurrence = True
        else:
            prob, size, forecast_qty, _ = _tsb(history, alpha=alpha, beta=beta)
            predicted_occurrence = prob >= threshold

        hits.append(predicted_occurrence == actual_occurred)
        if actual_occurred:
            magnitude_errors.append(abs(forecast_qty - actual_next) / actual_next * 100)

    if not hits:
        return None
    hit_rate = round(100 * sum(hits) / len(hits), 1)
    mag_err = round(sum(magnitude_errors) / len(magnitude_errors), 1) if magnitude_errors else None
    return hit_rate, mag_err


def tune_forecast_parameters(monthly_demand: pd.DataFrame, pattern_classification: pd.DataFrame,
                               min_history: int = 6) -> pd.DataFrame:
    """
    Per (project, part_code): grid-search alpha (and beta/threshold for
    Intermittent/Lumpy parts) against the walk-forward backtest, using a
    CONSTRAINED selection rule:
        1. Only consider settings whose occurrence_hit_rate_pct is >= the
           naive baseline.
        2. Among settings that clear that bar, keep whichever has the
           LOWEST magnitude error.
        3. If NOTHING clears the baseline, fall back to the single
           closest-to-baseline setting, tie-broken by magnitude error, and
           flag it clearly as `met_baseline = False`.
    """
    pattern_lookup = pattern_classification.set_index(["project", "part_code"])["demand_pattern"].to_dict()
    alpha_grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    beta_grid = [0.05, 0.1, 0.15, 0.2, 0.3]
    threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    records = []

    for (project, part_code), g in monthly_demand.groupby(["project", "part_code"]):
        g = g.sort_values("month")
        values = g["qty"].to_numpy(dtype=float)
        pattern = pattern_lookup.get((project, part_code), "Unknown")
        if len(values) < min_history + 2:
            continue

        baseline = _naive_baseline_hit_rate(values, min_history)
        all_results = []

        if pattern in ("Smooth", "Erratic"):
            for alpha in alpha_grid:
                result = _backtest_one_config(values, pattern, min_history, alpha=alpha)
                if result is not None:
                    all_results.append((*result, alpha, None, None))
        else:
            for alpha in alpha_grid:
                for beta in beta_grid:
                    for threshold in threshold_grid:
                        result = _backtest_one_config(values, pattern, min_history, alpha=alpha, beta=beta, threshold=threshold)
                        if result is not None:
                            all_results.append((*result, alpha, beta, threshold))

        if not all_results:
            continue

        qualifying = [r for r in all_results if baseline is None or r[0] >= baseline]
        met_baseline = len(qualifying) > 0
        pool = qualifying if met_baseline else all_results

        best = min(pool, key=lambda r: r[1] if r[1] is not None else float("inf"))
        hit_rate, mag_err, alpha, beta, threshold = best

        records.append({
            "project": project,
            "part_code": part_code,
            "demand_pattern": pattern,
            "months_of_history": len(values),
            "naive_baseline_hit_rate_pct": baseline,
            "tuned_occurrence_hit_rate_pct": hit_rate,
            "met_baseline": met_baseline,
            "tuned_avg_magnitude_error_pct": mag_err,
            "tuned_alpha": alpha,
            "tuned_beta": beta,
            "tuned_threshold": threshold,
        })

    if not records:
        return pd.DataFrame(columns=["project", "part_code", "demand_pattern", "months_of_history",
                                       "naive_baseline_hit_rate_pct", "tuned_occurrence_hit_rate_pct",
                                       "met_baseline", "tuned_avg_magnitude_error_pct",
                                       "tuned_alpha", "tuned_beta", "tuned_threshold"])
    return pd.DataFrame(records).sort_values("months_of_history", ascending=False)


def reconcile_forecast_log(forecast_log: pd.DataFrame, monthly_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the IMMUTABLE forecast log (real predictions made at the time,
    never rewritten) against actual monthly demand, wherever the actual
    outcome for that target month now exists. This is the real, answerable
    "were we right" record - not a retrospective simulation like
    backtest_accuracy, but what the model actually said before the answer
    was known.
    """
    if len(forecast_log) == 0:
        return pd.DataFrame(columns=["project", "part_code", "run_date", "target_month", "months_ahead",
                                       "predicted_to_occur", "occurrence_probability", "forecast_qty",
                                       "actual_qty", "actual_occurred", "correct_occurrence", "magnitude_error_pct"])

    actuals = monthly_demand.copy()
    actuals["month_str"] = actuals["month"].astype(str)
    actual_lookup = actuals.set_index(["project", "part_code", "month_str"])["qty"].to_dict()

    records = []
    for _, row in forecast_log.iterrows():
        key = (row["project"], row["part_code"], row["forecast_month"])
        if key not in actual_lookup:
            continue  # actual outcome for this target month doesn't exist yet - nothing to reconcile
        actual_qty = actual_lookup[key]
        actual_occurred = actual_qty > 0
        correct_occurrence = (row["predicted_to_occur"] == actual_occurred) if pd.notna(row["predicted_to_occur"]) else None
        magnitude_error_pct = round(abs(row["forecast_qty"] - actual_qty) / actual_qty * 100, 1) if actual_occurred else None

        records.append({
            "project": row["project"], "part_code": row["part_code"], "run_date": row["run_date"],
            "target_month": row["forecast_month"], "months_ahead": row["months_ahead"],
            "predicted_to_occur": row["predicted_to_occur"], "occurrence_probability": row["occurrence_probability"],
            "forecast_qty": row["forecast_qty"], "actual_qty": actual_qty, "actual_occurred": actual_occurred,
            "correct_occurrence": correct_occurrence, "magnitude_error_pct": magnitude_error_pct,
        })

    if not records:
        return pd.DataFrame(columns=["project", "part_code", "run_date", "target_month", "months_ahead",
                                       "predicted_to_occur", "occurrence_probability", "forecast_qty",
                                       "actual_qty", "actual_occurred", "correct_occurrence", "magnitude_error_pct"])
    return pd.DataFrame(records)


def summarize_forecast_reconciliation(reconciliation: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the reconciliation log into a real, live-tracked accuracy
    score per (project, part_code, months_ahead) - "how good are we at
    predicting 1 month out vs 6 months out," using only genuine logged
    predictions, not a simulation.
    """
    if len(reconciliation) == 0:
        return pd.DataFrame(columns=["project", "part_code", "months_ahead", "num_reconciled",
                                       "live_occurrence_hit_rate_pct", "live_avg_magnitude_error_pct"])
    records = []
    for (project, part_code, months_ahead), g in reconciliation.groupby(["project", "part_code", "months_ahead"]):
        valid_occ = g["correct_occurrence"].dropna()
        hit_rate = round(100 * valid_occ.mean(), 1) if len(valid_occ) > 0 else None
        mag_errs = g["magnitude_error_pct"].dropna()
        avg_mag_err = round(mag_errs.mean(), 1) if len(mag_errs) > 0 else None
        records.append({
            "project": project, "part_code": part_code, "months_ahead": months_ahead,
            "num_reconciled": len(g), "live_occurrence_hit_rate_pct": hit_rate,
            "live_avg_magnitude_error_pct": avg_mag_err,
        })
    return pd.DataFrame(records).sort_values(["part_code", "months_ahead"])
