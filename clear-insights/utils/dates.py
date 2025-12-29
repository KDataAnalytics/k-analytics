import pandas as pd


def looks_like_id_column(s: pd.Series, name: str) -> bool:
    """Heuristics to exclude ID/index columns from date auto-detection."""
    name_lower = str(name).lower()

    # Name-based ID/index patterns
    if any(x in name_lower for x in ["row id", "row_id", "index", "idx", "record", "uuid"]):
        return True

    # Generic "*id*" names are usually identifiers (unless they also look like date keys)
    if "id" in name_lower and not any(x in name_lower for x in ["date", "day", "month", "time", "timestamp", "key"]):
        return True

    # Sequential-ish integer columns with very high uniqueness are often IDs
    if pd.api.types.is_integer_dtype(s):
        s2 = s.dropna()
        if len(s2) == 0:
            return False

        uniq_ratio = s2.nunique() / len(s2)
        if uniq_ratio > 0.98:
            vals = s2.sort_values().unique()
            if len(vals) >= 10:
                diffs = pd.Series(vals).diff().dropna()
                if (diffs > 0).all() and diffs.median() == 1:
                    return True

    return False


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Robust date parsing:
    - If numeric and looks like YYYYMMDD (e.g., 20241109), parse with format.
    - Otherwise try standard parsing.
    """
    if pd.api.types.is_numeric_dtype(s):
        dt = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        if dt.notna().mean() >= 0.90:
            return dt

    return pd.to_datetime(s, errors="coerce")


def score_date_column(s: pd.Series) -> float:
    """Score by % parseable as dates."""
    parsed = parse_date_series(s)
    return float(parsed.notna().mean())


def score_value_column(s: pd.Series) -> float:
    """Prefer columns with good coverage + enough variation to forecast."""
    s2 = pd.to_numeric(s, errors="coerce")
    non_null = float(s2.notna().mean())
    variability = float(s2.std() / (s2.mean() + 1e-9)) if s2.notna().any() else 0.0
    return non_null + variability


def infer_frequency(dates: pd.Series) -> str:
    """Infer typical spacing between dates."""
    d = dates.sort_values()
    deltas = d.diff().dt.days.dropna()
    if deltas.empty:
        return "Daily"
    median = deltas.median()
    if median <= 1:
        return "Daily"
    if median <= 7:
        return "Weekly"
    return "Monthly"
