"""Home page view for the Streamlit application.

This module provides the home page view for the application, which includes
an introduction to the project, overview of the dataset, and key navigation
guidance for first-time users.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st


def _format_time_period(df: pd.DataFrame) -> str:
    """Return the visible time window covered by the processed dataset."""
    if "year_month" in df.columns:
        series = pd.to_datetime(df["year_month"], errors="coerce").dropna()
    elif "Tranc_YearMonth" in df.columns:
        series = pd.to_datetime(df["Tranc_YearMonth"], errors="coerce").dropna()
    elif {"Tranc_Year", "Tranc_Month"}.issubset(df.columns):
        series = pd.to_datetime(
            df["Tranc_Year"].astype(str)
            + "-"
            + df["Tranc_Month"].astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        ).dropna()
    else:
        series = pd.Series(dtype="datetime64[ns]")

    if series.empty:
        return "N/A"

    return f"{series.min():%Y}-{series.max():%Y}"


def _format_price_range(df: pd.DataFrame) -> str:
    """Return a compact price range string that fits comfortably in a metric."""
    if "resale_price" not in df.columns or df["resale_price"].dropna().empty:
        return "N/A"

    low = float(df["resale_price"].min())
    high = float(df["resale_price"].max())

    def compact(value: float) -> str:
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value / 1_000:.0f}k"

    return f"{compact(low)}-{compact(high)}"


@st.cache_data
def get_dataset_stats():
    """Load summary statistics for the home-page market snapshot."""
    try:
        root_dir = Path(__file__).parent.parent.parent
        data_path = os.path.join(root_dir, "data", "processed", "train_processed_exploratory.csv")

        if not os.path.exists(data_path):
            return {
                "transactions": "N/A",
                "time_period": "N/A",
                "towns": "N/A",
                "price_range": "N/A",
            }

        df = pd.read_csv(data_path, low_memory=False)

        stats = {
            "transactions": f"{len(df):,}",
            "time_period": _format_time_period(df),
            "towns": f"{df['town'].nunique()}" if "town" in df.columns else "N/A",
            "price_range": _format_price_range(df),
        }

        if "town" in df.columns:
            st.session_state["towns"] = sorted(df["town"].unique().tolist())

        if "flat_type" in df.columns:
            st.session_state["flat_types"] = sorted(df["flat_type"].unique().tolist())

        return stats

    except Exception as exc:
        st.error(f"Error loading dataset statistics: {exc}")
        return {
            "transactions": "Error",
            "time_period": "Error",
            "towns": "Error",
            "price_range": "Error",
        }


def show_home():
    """Display the home page content."""
    st.markdown("<h1 class='main-header'>HDB Resale Price Prediction</h1>", unsafe_allow_html=True)

    intro_col, guide_col = st.columns([1.8, 1.1], gap="large")
    with intro_col:
        st.markdown(
            """
            Use historical resale transactions to understand price levels across towns,
            compare flat types, and estimate a single flat price from practical buyer inputs.
            """
        )
    with guide_col:
        st.info(
            "Start here\n\n"
            "- Scan the market snapshot below.\n"
            "- Use Data Explorer for town and flat-type patterns.\n"
            "- Use Make Prediction when you already know the flat details."
        )

    action_cols = st.columns(3)
    action_cols[0].markdown(
        """
        **Data Explorer**

        Compare towns, flat types, and time periods before looking at a single estimate.
        """
    )
    action_cols[1].markdown(
        """
        **Make Prediction**

        Enter flat attributes and get a point estimate using the trained pricing models.
        """
    )
    action_cols[2].markdown(
        """
        **Model Performance**

        Inspect fit quality, feature impact, and error metrics before trusting a prediction.
        """
    )

    st.markdown("---")
    st.markdown("## Market Snapshot")

    stats = get_dataset_stats()

    cols = st.columns(4)
    with cols[0]:
        st.metric("Transactions", stats["transactions"])
    with cols[1]:
        st.metric("Coverage", stats["time_period"])
    with cols[2]:
        st.metric("Towns", stats["towns"])
    with cols[3]:
        st.metric("Price Range", stats["price_range"])

    st.markdown("---")
    st.markdown("## What powers the estimate")
    detail_cols = st.columns(2, gap="large")
    detail_cols[0].markdown(
        """
        - Historical resale transactions from `data.gov.sg`
        - Location, flat type, floor area, lease, and amenity-distance features
        - Separate views for market exploration and single-flat prediction
        """
    )
    detail_cols[1].markdown(
        """
        - Linear, ridge, and lasso regression baselines
        - Model quality reported with `R^2` and RMSE
        - Prediction inputs focused on buyer-known property details
        """
    )
