import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.icons import heading_html
from ui.navigation import back_button


# ------------------------------------------------------------
# STEP 1 - PROFILE DATA
# ------------------------------------------------------------
def _detect_boolean_columns(df: pd.DataFrame) -> list[str]:
    bool_values = {"yes", "y", "true", "t", "1", "no", "n", "false", "f", "0"}
    candidates = []
    for col in df.columns:
        if df[col].dtype != object and df[col].dtype.name != "string":
            continue
        values = df[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        if values and set(values).issubset(bool_values):
            candidates.append(col)
    return candidates


def _detect_date_columns(df: pd.DataFrame, threshold: float = 0.7) -> list[str]:
    return [col for col, rate in _date_parse_rates(df).items() if rate >= threshold]


def _date_parse_rates(df: pd.DataFrame) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            continue
        sample = df[col].dropna()
        if sample.empty:
            rates[col] = 0.0
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        rates[col] = float(parsed.notna().mean())
    return rates


def _detect_string_cleanup(df: pd.DataFrame) -> dict:
    issues = {"whitespace": [], "case": []}
    for col in df.columns:
        if df[col].dtype != object and df[col].dtype.name != "string":
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        if (series.str.strip() != series).any():
            issues["whitespace"].append(col)
        lowered = series.str.strip().str.lower()
        if lowered.nunique() < series.nunique():
            issues["case"].append(col)
    return issues


def _sample_values(series: pd.Series, limit: int = 4) -> str:
    values = series.dropna().astype(str).unique().tolist()
    if not values:
        return "-"
    sample = values[:limit]
    suffix = "..." if len(values) > limit else ""
    return ", ".join(sample) + suffix


def _apply_cleaning(
    df: pd.DataFrame,
    remove_duplicates: bool,
    trim_case: bool,
    normalize_bools: bool,
    date_cols_to_parse: list[str],
    fill_missing: bool,
    drop_missing_cols: bool,
    drop_cols: list[str]
) -> tuple[pd.DataFrame, dict]:
    changes = {}
    cleaned = df.copy()

    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        changes["duplicates_removed"] = before - len(cleaned)

    if trim_case:
        string_issues = _detect_string_cleanup(cleaned)
        string_cols = sorted(set(string_issues["whitespace"] + string_issues["case"]))
        for col in string_cols:
            cleaned[col] = cleaned[col].astype("string").str.strip().str.title()
        changes["string_standardized"] = len(string_cols)

    if normalize_bools:
        bool_cols = _detect_boolean_columns(cleaned)
        mapping = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0
        }
        for col in bool_cols:
            cleaned[col] = cleaned[col].map(lambda x: mapping.get(str(x).strip().lower(), x))
        changes["boolean_normalized"] = len(bool_cols)

    if date_cols_to_parse:
        for col in date_cols_to_parse:
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
        changes["dates_parsed"] = len(date_cols_to_parse)

    if drop_missing_cols:
        cleaned = cleaned.drop(columns=drop_cols, errors="ignore")
        changes["columns_dropped"] = len(drop_cols)

    if fill_missing:
        missing_before = int(cleaned.isna().sum().sum())
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cat_cols = cleaned.select_dtypes(include=["object", "string"]).columns
        for col in num_cols:
            median = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median)
        for col in cat_cols:
            cleaned[col] = cleaned[col].fillna("Unknown")
        changes["missing_filled"] = missing_before

    return cleaned, changes


def step_1_profile() -> None:
    df = st.session_state.df
    if "df_original" not in st.session_state:
        st.session_state.df_original = df.copy()

    st.markdown(heading_html("Step 2: Profile Your Data", "chart", level=1), unsafe_allow_html=True)
    st.markdown("Here is a summary of your uploaded dataset:")

    if "data_prep_changes" in st.session_state:
        st.success(f"Cleaning applied: {st.session_state.data_prep_changes}")
        del st.session_state.data_prep_changes

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing values", df.isna().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())

    missing_ratio = df.isna().mean()
    missing_cols = missing_ratio[missing_ratio > 0].index.tolist()
    high_missing_cols = missing_ratio[missing_ratio > 0.4].index.tolist()
    bool_cols = _detect_boolean_columns(df)
    date_rates = _date_parse_rates(df)
    date_cols = [c for c, r in date_rates.items() if r >= 0.7]
    mixed_date_cols = [c for c, r in date_rates.items() if 0.2 <= r < 0.7]
    string_issues = _detect_string_cleanup(df)

    issues = {
        "duplicates": df.duplicated().sum() > 0,
        "missing": len(missing_cols) > 0,
        "high_missing": len(high_missing_cols) > 0,
        "bools": len(bool_cols) > 0,
        "dates": len(date_cols) > 0,
        "strings": len(string_issues["whitespace"]) > 0 or len(string_issues["case"]) > 0,
        "mixed_dates": len(mixed_date_cols) > 0
    }

    with st.expander("Show data quality details", expanded=False):
        if not any(issues.values()):
            st.success("No data quality issues detected.")
        if issues["duplicates"]:
            st.info(f"Found {df.duplicated().sum():,} duplicate rows.")
        if issues["missing"]:
            st.info(f"Missing values in {len(missing_cols)} columns.")
        if issues["high_missing"]:
            st.warning(f"{len(high_missing_cols)} columns exceed 40% missing values.")
        if issues["strings"]:
            st.info("Text columns show inconsistent capitalization or extra whitespace.")
        if issues["bools"]:
            st.info(f"{len(bool_cols)} columns look like boolean values.")
        if issues["dates"]:
            st.info(f"{len(date_cols)} columns look like date values.")
        if issues["mixed_dates"]:
            st.warning(f"{len(mixed_date_cols)} columns have mixed date-like values.")

    if not any(issues.values()):
        st.success("Your dataset looks clean. You're ready to move on.")
    else:
        st.markdown(heading_html("Flagged Columns", "alert", level=3), unsafe_allow_html=True)
        flagged_rows = []
        for col in df.columns:
            notes = []
            if col in string_issues["whitespace"]:
                notes.append("extra whitespace")
            if col in string_issues["case"]:
                notes.append("mixed capitalization")
            if col in bool_cols:
                notes.append("boolean-like values")
            if col in date_cols:
                notes.append("date-like values")
            if col in mixed_date_cols:
                notes.append("mixed date-like values")
            if col in high_missing_cols:
                notes.append("high missing %")
            if col in missing_cols:
                notes.append("missing values")
            if notes:
                flagged_rows.append({
                    "Column": col,
                    "Issues": ", ".join(sorted(set(notes))),
                    "Sample Values": _sample_values(df[col])
                })
        if flagged_rows:
            st.dataframe(pd.DataFrame(flagged_rows), use_container_width=True)

        st.markdown(heading_html("Recommended Fixes", "target", level=3), unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fix_duplicates = st.checkbox("Remove duplicate rows", value=issues["duplicates"])
            fix_strings = st.checkbox("Trim + standardize capitalization", value=issues["strings"])
            fix_bools = st.checkbox("Normalize boolean values", value=issues["bools"])
        with col_b:
            fix_dates = st.checkbox("Parse date columns", value=issues["dates"])
            fix_missing = st.checkbox("Fill missing values (median/Unknown)", value=issues["missing"])
            fix_drop = st.checkbox("Drop columns with high missing %", value=False)

        date_cols_to_parse = []
        if fix_dates:
            default_dates = date_cols
            date_cols_to_parse = st.multiselect(
                "Select columns to parse as dates",
                options=sorted(set(date_cols + mixed_date_cols)),
                default=default_dates
            )
            if mixed_date_cols:
                st.caption("Columns with mixed date-like values are included but not selected by default.")

        drop_cols = []
        if fix_drop:
            missing_threshold_pct = st.slider(
                "High-missing threshold",
                min_value=20,
                max_value=90,
                value=40,
                step=5,
                format="%d%%"
            )
            high_missing_cols_dynamic = missing_ratio[missing_ratio > (missing_threshold_pct / 100)].index.tolist()
            drop_cols = st.multiselect(
                "Select columns to drop",
                options=sorted(set(high_missing_cols_dynamic)),
                default=high_missing_cols_dynamic
            )
            if not drop_cols:
                st.caption("No columns selected for dropping.")

        if st.button("Apply selected fixes", type="primary"):
            cleaned, changes = _apply_cleaning(
                df,
                remove_duplicates=fix_duplicates,
                trim_case=fix_strings,
                normalize_bools=fix_bools,
                date_cols_to_parse=date_cols_to_parse,
                fill_missing=fix_missing,
                drop_missing_cols=fix_drop,
                drop_cols=drop_cols
            )
            st.session_state.df = cleaned
            st.session_state.data_prep_changes = changes
            st.session_state.df_cleaned = cleaned
            st.rerun()

    st.markdown(heading_html("Preview", "search", level=3), unsafe_allow_html=True)
    if "df_cleaned" in st.session_state:
        before, after = st.tabs(["Before", "After"])
        with before:
            st.dataframe(st.session_state.df_original.head(20), use_container_width=True)
        with after:
            st.dataframe(df.head(20), use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.dataframe(df.head(20), use_container_width=True)

    numeric = df.select_dtypes(include=[np.number])
    st.markdown(heading_html("Correlation (Numeric Columns Only)", "trend_up", level=3), unsafe_allow_html=True)
    if len(numeric.columns) >= 2:
        fig = px.imshow(
            numeric.corr().round(2),
            text_auto=True,
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

    st.markdown("---")
    if st.button("Continue -> Select Objective", type="primary"):
        st.session_state.step = 2
        st.rerun()

    # <- Back button to return to file upload
    back_button(0)
