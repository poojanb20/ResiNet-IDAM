
import pandas as pd
import streamlit as st

from pathlib import Path

# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="ResiNet-IDAM",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = (
    BASE_DIR
    / "reports"
    / "it"
    / "ids2025_zero_trust_results.csv"
)

st.title("🛡️ ResiNet-IDAM")
st.subheader("IT Intrusion Detection & Zero Trust Risk Dashboard")

st.caption(
    "Research prototype | IDS2025 dataset | "
    "Autoencoder anomaly detection + rule-based risk scoring"
)

# --------------------------------------------------
# Load results
# --------------------------------------------------

@st.cache_data
def load_results(path, modified_time):
    return pd.read_csv(path)

if not RESULTS_PATH.exists():
    st.error(f"Results file not found: {RESULTS_PATH}")
    st.info(
        "Run `python scripts/it/integrate_detection_risk.py` "
        "from your project directory first."
    )
    st.stop()

df = load_results(
    str(RESULTS_PATH),
    RESULTS_PATH.stat().st_mtime,
)

required_columns = {
    "newLabel",
    "reconstruction_error",
    "anomaly_score",
    "risk_score",
    "risk_level",
    "recommended_action",
}

missing = required_columns - set(df.columns)

if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filters")

risk_options = ["LOW", "MEDIUM", "HIGH"]
selected_risks = st.sidebar.multiselect(
    "Risk level",
    options=risk_options,
    default=risk_options,
)

labels = sorted(df["newLabel"].dropna().unique().tolist())

selected_labels = st.sidebar.multiselect(
    "Traffic class",
    options=labels,
    default=labels,
)

minimum_risk, maximum_risk = st.sidebar.slider(
    "Risk score range",
    min_value=0,
    max_value=100,
    value=(0, 100),
)

search_text = st.sidebar.text_input(
    "Search recommended action",
    placeholder="e.g. RESTRICT",
)

filtered = df[
    df["risk_level"].isin(selected_risks)
    & df["newLabel"].isin(selected_labels)
    & df["risk_score"].between(minimum_risk, maximum_risk)
].copy()

if search_text:
    filtered = filtered[
        filtered["recommended_action"].str.contains(
            search_text,
            case=False,
            na=False,
        )
    ]

# --------------------------------------------------
# Overview metrics
# --------------------------------------------------

st.header("Security Overview")

total_records = len(df)
high_risk = int((df["risk_level"] == "HIGH").sum())
medium_risk = int((df["risk_level"] == "MEDIUM").sum())
low_risk = int((df["risk_level"] == "LOW").sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total records", f"{total_records:,}")
c2.metric("High risk", f"{high_risk:,}")
c3.metric("Medium risk", f"{medium_risk:,}")
c4.metric("Low risk", f"{low_risk:,}")

st.caption(
    "Risk levels are prototype classifications. "
    "They do not confirm that an individual record is malicious."
)

# --------------------------------------------------
# Charts
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Risk Distribution")
    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
    )
    st.bar_chart(risk_counts)

with right:
    st.subheader("Traffic Classes")
    class_counts = df["newLabel"].value_counts()
    st.bar_chart(class_counts)

# --------------------------------------------------
# Filtered records
# --------------------------------------------------

st.header("Risk Assessment Records")

st.write(
    f"Showing **{len(filtered):,}** of **{len(df):,}** records."
)

display_columns = [
    "newLabel",
    "reconstruction_error",
    "anomaly_score",
    "asset_criticality",
    "trust_score",
    "risk_score",
    "risk_level",
    "recommended_action",
]

display_columns = [
    col for col in display_columns if col in filtered.columns
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# Selected-record details
# --------------------------------------------------

st.header("Record Investigation")

if not filtered.empty:
    row_number = st.selectbox(
        "Select a displayed record",
        options=list(range(len(filtered))),
        format_func=lambda i: (
            f"Record {i + 1} | "
            f"{filtered.iloc[i]['newLabel']} | "
            f"Risk {filtered.iloc[i]['risk_score']:.2f}"
        ),
    )

    selected = filtered.iloc[row_number]

    d1, d2, d3 = st.columns(3)
    d1.metric("Risk score", f"{selected['risk_score']:.2f}/100")
    d2.metric("Risk level", str(selected["risk_level"]))
    d3.metric(
        "Anomaly score",
        f"{selected['anomaly_score']:.4f}",
    )

    st.write("**Recommended action:**", selected["recommended_action"])
    st.write("**Traffic class:**", selected["newLabel"])
    st.write(
        "**Reconstruction error:**",
        f"{selected['reconstruction_error']:.6f}",
    )
else:
    st.info("No records match the selected filters.")

# --------------------------------------------------
# Export
# --------------------------------------------------

st.header("Export")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered records as CSV",
    data=csv_data,
    file_name="resinet_idam_filtered_results.csv",
    mime="text/csv",
)

st.caption(
    "Prototype only: no live traffic monitoring or automatic "
    "access enforcement is implemented in this dashboard."
)
