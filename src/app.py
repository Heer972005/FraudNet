import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from streamlit_autorefresh import st_autorefresh

from graph_detector import (
    build_relationship_graph,
    detect_communities,
    rank_communities,
    calculate_community_stats
)

from live_simulator import generate_live_transaction
from report_generator import generate_report

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="FraudNet",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------

accounts = pd.read_csv("data/raw/accounts.csv")
transactions = pd.read_csv("data/raw/transactions.csv")
refunds = pd.read_csv("data/raw/refunds.csv")

# -----------------------------
# Session State
# -----------------------------

if "live_feed" not in st.session_state:
    st.session_state.live_feed = []

# -----------------------------
# Build Fraud Graph
# -----------------------------

G = build_relationship_graph(
    accounts,
    transactions,
    refunds
)

communities = detect_communities(G)

ranked = rank_communities(
    G,
    communities,
    accounts
)

community_stats = calculate_community_stats(
    G,
    communities,
    accounts
)

# -----------------------------
# Business Metrics
# -----------------------------

total_accounts = len(accounts)
total_transactions = len(transactions)

high_risk = [
    c for c in ranked
    if c["risk_score"] >= 80
]

estimated_savings = len(high_risk) * 5000

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🔍 Risk Investigation")

risk_threshold = st.sidebar.slider(
    "Minimum Community Risk",
    0,
    100,
    60
)

filtered = [
    c for c in ranked
    if c["risk_score"] >= risk_threshold
]

if not filtered:
    st.sidebar.warning("No communities found.")
    st.stop()

selected = st.sidebar.selectbox(
    "Choose Community",
    [
        f"Community {c['community_id']} (Risk {c['risk_score']})"
        for c in filtered
    ]
)

selected_id = int(selected.split()[1])

selected_stat = next(
    s for s in community_stats
    if s["community_id"] == selected_id
)

selected_info = next(
    c for c in filtered
    if c["community_id"] == selected_id
)

members = selected_stat["members"]

# -----------------------------
# Header
# -----------------------------

st.title("🛡️ FraudNet")

st.caption(
    "AI Risk Manager that detects coordinated fraud rings, explains every decision, and helps merchants reduce refund and chargeback losses."
)

# -----------------------------
# KPI Cards
# -----------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("👥 Accounts", f"{total_accounts:,}")
c2.metric("💳 Transactions", f"{total_transactions:,}")
c3.metric("🚨 High-Risk Communities", len(high_risk))
c4.metric("💰 Potential Savings", f"₹{estimated_savings:,}")

st.divider()

# -----------------------------
# Merchant Action Center
# -----------------------------

highest = ranked[0]

st.subheader("🚨 Merchant Action Center")

st.error(
    f"""
### Community {highest['community_id']} requires immediate review

**Risk Score:** {highest['risk_score']}/100

**Accounts:** {highest['size']}

**Recommended Action:** Pause automatic refunds and perform manual verification.
"""
)

x1, x2, x3 = st.columns(3)

x1.info("🔗 Shared devices detected")
x2.info("💳 Payment fingerprint reuse")
x3.info("🆕 Recently created accounts")

# -----------------------------
# Live Payment Monitor
# -----------------------------

st.divider()

st.subheader("⚡ Live Payment Monitor")

st.caption("Incoming merchant payments are evaluated every 3 seconds.")

st_autorefresh(
    interval=3000,
    key="fraudnet_refresh"
)

txn = generate_live_transaction(
    accounts,
    members
)

st.session_state.live_feed.insert(0, txn)

st.session_state.live_feed = st.session_state.live_feed[:10]

for txn in st.session_state.live_feed:

    if "Fraud Alert" in txn["status"]:

        st.error(
            f"""
**{txn['time']} • {txn['merchant']}**

Account: {txn['account_id']}

Amount: ₹{txn['amount']:,}

Status: {txn['status']}
"""
        )

        st.chat_message("assistant").write(
            f"""
FraudNet automatically investigated this payment.

- Connected to Community {selected_id}
- Shared device signals detected
- Coordinated behavior observed

**Recommendation:** Manual Review
"""
        )

    elif "Review" in txn["status"]:

        st.warning(
            f"""
**{txn['time']} • {txn['merchant']}**

Account: {txn['account_id']}

Amount: ₹{txn['amount']:,}

Status: {txn['status']}
"""
        )

    else:

        st.success(
            f"""
**{txn['time']} • {txn['merchant']}**

Account: {txn['account_id']}

Amount: ₹{txn['amount']:,}

Status: {txn['status']}
"""
        )

# -----------------------------
# Risk Communities Table
# -----------------------------

st.divider()

st.subheader("📋 Communities Requiring Attention")

table = pd.DataFrame(filtered)[
    ["community_id", "risk_score", "size", "reasons"]
]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Merchant Copilot
# -----------------------------

st.divider()

st.subheader("💬 Merchant Copilot")

questions = [
    "Why was this community flagged?",
    "Should I block these accounts?",
    "Summarize this investigation.",
    "Write a chargeback response."
]

selected_question = st.selectbox(
    "Quick Questions",
    questions
)

if st.button("Ask FraudNet"):

    risk = selected_info["risk_score"]
    size = selected_stat["size"]
    devices = selected_stat["shared_devices"]
    age = round(selected_stat["avg_account_age"])
    reasons = selected_info["reasons"]

    if selected_question == "Why was this community flagged?":

        answer = f"""
Community {selected_id} scored **{risk}/100** because FraudNet detected:

- {size} connected accounts
- {devices} shared devices
- Average account age of {age} days
- Indicators: {reasons}
"""

    elif selected_question == "Should I block these accounts?":

        answer = f"""
Don't immediately block them.

Recommended workflow:

1. Pause refunds.
2. Request verification.
3. Monitor future transactions.
"""

    elif selected_question == "Summarize this investigation.":

        answer = f"""
Community {selected_id} is a high-risk cluster containing {size} linked accounts with a fraud score of {risk}/100.
"""

    else:

        answer = f"""
Subject: Chargeback Investigation

Community {selected_id} was identified as a coordinated fraud cluster.

Risk Score: {risk}/100

Recommendation: Review before approving refunds.
"""

    st.chat_message("user").write(selected_question)
    st.chat_message("assistant").write(answer)

# -----------------------------
# Business Insights
# -----------------------------

st.divider()

st.subheader("📊 Business Insights")

left, right = st.columns(2)

with left:

    distribution = px.histogram(
        accounts,
        x="account_type",
        title="Account Distribution"
    )

    st.plotly_chart(
        distribution,
        use_container_width=True,
        key="distribution"
    )

with right:

    refund_counts = (
        refunds.groupby("account_id")
        .size()
        .reset_index(name="refunds")
    )

    merged = accounts.merge(
        refund_counts,
        on="account_id",
        how="left"
    )

    merged["refunds"] = merged["refunds"].fillna(0)

    refund_chart = px.box(
        merged,
        x="account_type",
        y="refunds",
        title="Refund Behavior"
    )

    st.plotly_chart(
        refund_chart,
        use_container_width=True,
        key="refund_chart"
    )

# -----------------------------
# Selected Community
# -----------------------------

st.divider()

st.subheader("🔍 Selected Community")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Risk", selected_info["risk_score"])
m2.metric("Members", selected_stat["size"])
m3.metric("Shared Devices", selected_stat["shared_devices"])
m4.metric(
    "Avg Account Age",
    f"{round(selected_stat['avg_account_age'])} days"
)

# -----------------------------
# Network Graph
# -----------------------------

st.markdown(
    """
### 🕸 Relationship Network

- 🔴 Fraud
- 🟠 Household
- 🔵 Normal
"""
)

subgraph = G.subgraph(members)

sub_pos = nx.spring_layout(
    subgraph,
    seed=42
)

edge_x = []
edge_y = []

for s, t in subgraph.edges():

    x0, y0 = sub_pos[s]
    x1, y1 = sub_pos[t]

    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    hoverinfo="none"
)

lookup = accounts.set_index("account_id")

node_x = []
node_y = []
colors = []
sizes = []
hover = []

for node in subgraph.nodes():

    x, y = sub_pos[node]

    node_x.append(x)
    node_y.append(y)

    account = lookup.loc[node]

    degree = subgraph.degree(node)

    sizes.append(12 + degree)

    if account["is_fraud_ring"]:
        colors.append("red")
    elif account["account_type"] == "household":
        colors.append("orange")
    else:
        colors.append("steelblue")

    hover.append(
        f"{node}<br>{account['account_type']}<br>Connections: {degree}"
    )

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    text=hover,
    hovertemplate="%{text}",
    marker=dict(
        color=colors,
        size=sizes,
        line=dict(width=1, color="white")
    )
)

network_fig = go.Figure(
    data=[edge_trace, node_trace]
)

network_fig.update_layout(
    showlegend=False,
    hovermode="closest",
    height=650,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(
    network_fig,
    use_container_width=True,
    key="network"
)

# -----------------------------
# Investigation Summary
# -----------------------------

st.divider()

st.subheader("📄 Investigation Summary")

for reason in selected_info["reasons"].split(", "):
    st.write(f"• {reason}")

st.warning(
    """
**FraudNet Recommendation**

- Pause automatic refunds.
- Request additional verification.
- Monitor linked payment fingerprints.
- Continue watching connected devices.
"""
)

# -----------------------------
# AI Report Generator
# -----------------------------

st.divider()

st.subheader("🤖 AI Investigation Report")

if st.button("Generate Investigation Report"):

    report = f"""
# Chargeback Investigation Report

Community {selected_id} has been identified as a high-risk coordinated account cluster.

Risk Score: {selected_info['risk_score']}/100

Connected Accounts: {selected_stat['size']}

Shared Devices: {selected_stat['shared_devices']}

Average Account Age: {round(selected_stat['avg_account_age'])} days

Why FraudNet Flagged This

{chr(10).join([f"- {r}" for r in selected_info["reasons"].split(", ")])}

Recommended Action

Pause automatic refunds.

Request manual verification.

Continue monitoring linked payment fingerprints and devices.
"""

    st.markdown(report)

# -----------------------------
# PDF Export
# -----------------------------

st.divider()

st.subheader("📄 Export Investigation Report")

if st.button("Generate PDF Report"):

    filename = f"FraudNet_Report_{selected_id}.pdf"

    generate_report(
        filename=filename,
        community_id=selected_id,
        risk=selected_info["risk_score"],
        members=selected_stat["size"],
        shared_devices=selected_stat["shared_devices"],
        avg_age=round(selected_stat["avg_account_age"]),
        reasons=selected_info["reasons"],
        savings=estimated_savings
    )

    with open(filename, "rb") as f:

        st.download_button(
            "⬇ Download Investigation Report",
            f,
            file_name=filename,
            mime="application/pdf"
        )

st.success("FraudNet dashboard is running successfully.")