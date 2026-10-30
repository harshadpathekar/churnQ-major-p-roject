import pandas as pd

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

NEW_TENURE_MAX = 6      # months
FEW_SERVICES_MAX = 1
MANY_SERVICES_MIN = 4

SEGMENTS = ["Premium", "At-Risk Loyal", "Churner - Price Sensitive",
            "Churner - Disengaged", "New Customer", "Stable Low-Value"]


def count_services(df):
    cols = [c for c in SERVICE_COLS if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index)
    return df[cols].apply(
        lambda s: s.astype(str).str.strip().str.lower().eq("yes")
    ).sum(axis=1)


def assign_segments(df):
    """df must have: tenure, MonthlyCharges, Contract, predicted_churn
    (Yes/No) and the service columns. Returns a copy with
    'num_services' and 'segment' added."""
    df = df.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["num_services"] = count_services(df)

    # thresholds come from THIS file's own distribution
    high_charge = df["MonthlyCharges"] >= df["MonthlyCharges"].quantile(0.75)
    long_tenure = df["tenure"] >= df["tenure"].quantile(0.50)
    new_cust = df["tenure"] <= NEW_TENURE_MAX
    churn = df["predicted_churn"].astype(str).str.lower().eq("yes")
    m2m = df["Contract"].astype(str).str.lower().str.contains("month")
    few = df["num_services"] <= FEW_SERVICES_MAX
    many = df["num_services"] >= MANY_SERVICES_MIN

    seg = pd.Series("Stable Low-Value", index=df.index)
    # assign lowest priority first so higher-priority rules overwrite
    seg[churn] = "Churner - Disengaged"               # fallback for churners
    seg[~churn & high_charge & long_tenure & many] = "Premium"
    seg[churn & few] = "Churner - Disengaged"
    seg[churn & long_tenure & high_charge] = "At-Risk Loyal"
    seg[churn & high_charge & m2m & ~long_tenure] = "Churner - Price Sensitive"
    seg[churn & new_cust] = "New Customer"

    df["segment"] = seg
    return df


def segment_summary(df):
    counts = df["segment"].value_counts().reindex(SEGMENTS, fill_value=0)
    by_risk = pd.crosstab(df["segment"], df["risk_level"]).reindex(SEGMENTS, fill_value=0)
    return counts, by_risk
