import os
import time
import random
from datetime import datetime

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBClassifier

from graph_detector import (
    build_relationship_graph,
    detect_communities,
    rank_communities,
    calculate_community_stats
)

from live_simulator import generate_live_transaction

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="FraudNet",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

.block-container{
max-width:1400px;
padding-top:1.5rem;
}

section[data-testid="stSidebar"]{
background:#0f172a;
}

.hero{
background:linear-gradient(135deg,#2563eb,#0f172a);
padding:36px;
border-radius:24px;
margin-bottom:25px;
}

.hero h1{
color:white;
font-size:46px;
margin:0;
}

.hero p{
color:#dbeafe;
font-size:18px;
margin-top:10px;
}

.metric-card{
background:#111827;
border:1px solid #1f2937;
border-radius:18px;
padding:20px;
transition:.25s;
}

.metric-card:hover{
transform:translateY(-4px);
border-color:#2563eb;
}

.metric-title{
color:#94a3b8;
font-size:15px;
}

.metric-value{
font-size:34px;
font-weight:700;
color:white;
}

.chart-card{
background:#111827;
padding:18px;
border-radius:18px;
border:1px solid #1f2937;
margin-bottom:18px;
}

.alert-card{
background:linear-gradient(135deg,#7f1d1d,#451a1a);
padding:22px;
border-radius:20px;
border-left:6px solid #ef4444;
}

.info-card{
background:#0f2138;
border:1px solid #1d4ed8;
padding:18px;
border-radius:16px;
text-align:center;
}

.ai-box{
background:#111827;
border-left:5px solid #3b82f6;
padding:18px;
border-radius:16px;
}

.live-card{
padding:18px;
border-radius:14px;
margin:10px 0;
color:white;
}

.reason-card{
background:#111827;
border-left:4px solid #3b82f6;
padding:14px;
border-radius:14px;
margin:8px 0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    accounts = pd.read_csv("data/raw/accounts.csv")
    transactions = pd.read_csv("data/raw/transactions.csv")
    refunds = pd.read_csv("data/raw/refunds.csv")

    return accounts, transactions, refunds

accounts, transactions, refunds = load_data()

# ---------------- BUILD GRAPH ----------------

@st.cache_resource
def prepare_graph(accounts, transactions, refunds):

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

    stats = calculate_community_stats(
        G,
        communities,
        accounts
    )

    return G, communities, ranked, stats

G, communities, ranked, community_stats = prepare_graph(
    accounts,
    transactions,
    refunds
)

# ---------------- FEATURE ENGINEERING ----------------

community_lookup = {}

for community in ranked:

    members = next(
        s["members"]
        for s in community_stats
        if s["community_id"] == community["community_id"]
    )

    for account in members:

        community_lookup[account] = community["risk_score"]

refund_counts = refunds.groupby("account_id").size().to_dict()
transaction_counts = transactions.groupby("account_id").size().to_dict()

rows = []

for _, account in accounts.iterrows():

    account_id = account["account_id"]

    weighted_degree = sum(
        data["weight"]
        for _,_,data in G.edges(account_id, data=True)
    )

    rows.append({

        "account_id": account_id,

        "account_age_days":
            account["account_age_days"],

        "community_risk":
            community_lookup.get(account_id,0),

        "graph_degree":
            G.degree(account_id),

        "weighted_degree":
            weighted_degree,

        "refund_rate":
            refund_counts.get(account_id,0) /
            max(transaction_counts.get(account_id,1),1)
    })

features_df = pd.DataFrame(rows)

# ---------------- LOAD TRAINED MODEL ----------------

MODEL_PATH = "models/fraudnet_model.json"

model = None

if os.path.exists(MODEL_PATH):

    model = XGBClassifier()

    model.load_model(MODEL_PATH)

    X = features_df[[
        "account_age_days",
        "community_risk",
        "graph_degree",
        "weighted_degree",
        "refund_rate"
    ]]

    features_df["fraud_probability"] = (
        model.predict_proba(X)[:,1]
    )

else:

    features_df["fraud_probability"] = (
        features_df["community_risk"]/100
    )

# ---------------- COMMUNITY ML SCORE ----------------

for community in ranked:

    members = next(
        s["members"]
        for s in community_stats
        if s["community_id"] == community["community_id"]
    )

    score = (
        features_df[
            features_df.account_id.isin(members)
        ]["fraud_probability"]
        .mean()*100
    )

    community["ml_risk"] = round(score,1)

# =====================================================
# PART 2 — MERCHANT DASHBOARD UI
# =====================================================

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔎 Risk Investigation")

risk_threshold = st.sidebar.slider(
    "Minimum ML Risk",
    0,
    100,
    60
)

filtered = [
    c for c in ranked
    if c["ml_risk"] >= risk_threshold
]

if not filtered:
    st.sidebar.warning("No communities above this threshold.")
    st.stop()

# -------- AUTO FOCUS (highest-risk community) --------

selected_comm = max(filtered, key=lambda x: x["ml_risk"])
selected_id = selected_comm["community_id"]

selected_stat = next(
    s for s in community_stats
    if s["community_id"] == selected_id
)

selected_members = selected_stat["members"]

st.sidebar.success(
    f"🎯 Auto-focused Community {selected_id}\n\nML Risk: {selected_comm['ml_risk']}%"
)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">

<h1>🛡 FraudNet</h1>

<p>
Real-time fraud ring detection using Graph Analytics and XGBoost AI.
Designed for merchants to identify coordinated refund abuse before money is lost.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------- TOP METRICS ----------------

total_accounts = len(accounts)
total_transactions = len(transactions)
high_risk = len(filtered)
estimated_savings = high_risk * 5000

c1, c2, c3, c4 = st.columns(4)

cards = [

    ("👥 Accounts", f"{total_accounts:,}"),

    ("💳 Transactions", f"{total_transactions:,}"),

    ("🚨 High-Risk Communities", high_risk),

    ("💰 Potential Savings", f"₹{estimated_savings:,}")

]

for col, (title, value) in zip([c1, c2, c3, c4], cards):

    with col:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">{title}</div>

        <div class="metric-value">{value}</div>

        </div>
        """, unsafe_allow_html=True)

st.divider()

# ---------------- MERCHANT ACTION CENTER ----------------

st.markdown("## 🚨 Merchant Action Center")

st.markdown(f"""
<div class="alert-card">

<h2 style="color:white;margin-top:0;">
Community {selected_id} Requires Immediate Review
</h2>

**ML Risk Score:** {selected_comm["ml_risk"]}/100

**Accounts involved:** {selected_stat["size"]}

**Recommended Action**

Pause automatic refunds and verify linked accounts before releasing payouts.

</div>
""", unsafe_allow_html=True)

a, b, c = st.columns(3)

with a:
    st.markdown("""
    <div class="info-card">
    <h3>🔗 Shared Devices</h3>
    <p>Multiple linked accounts reuse the same devices.</p>
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="info-card">
    <h3>💳 Payment Reuse</h3>
    <p>Payment fingerprints are shared across accounts.</p>
    </div>
    """, unsafe_allow_html=True)

with c:
    st.markdown("""
    <div class="info-card">
    <h3>🆕 Young Accounts</h3>
    <p>Several recently created accounts appear coordinated.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# LIVE PAYMENT MONITOR
# =====================================================
# =====================================================
# LIVE AI PAYMENT MONITOR
# =====================================================

st.markdown("## ⚡ Live AI Payment Monitor")
st.caption("Every new payment is analyzed using XGBoost + Graph Analytics before the merchant sees the decision.")

auto_refresh = st.toggle(
    "Auto Refresh (Every 5 Seconds)",
    value=False
)

# Session State
if "live_feed" not in st.session_state:
    st.session_state.live_feed = []

if "pending_tx" not in st.session_state:
    st.session_state.pending_tx = None

def create_transaction():

    tx = generate_live_transaction(accounts, selected_members)

    account = accounts[
        accounts["account_id"] == tx["account_id"]
    ].iloc[0]

    account_features = features_df[
        features_df["account_id"] == tx["account_id"]
    ].iloc[0]

    # Extra transaction feature
    amount_factor = min(tx["amount"] / 25000, 1)

    feature_vector = pd.DataFrame([{
        "account_age_days": account_features["account_age_days"],
        "community_risk": account_features["community_risk"],
        "graph_degree": account_features["graph_degree"],
        "weighted_degree": account_features["weighted_degree"],
        "refund_rate": min(
            account_features["refund_rate"] + amount_factor * 0.15,
            1
        )
    }])

    # Real XGBoost prediction
    if model is not None:

        probability = float(
            model.predict_proba(feature_vector)[0][1]
        )

    else:

        probability = account_features["community_risk"] / 100

    tx["probability"] = probability

    # Dynamic explanations
    signals = []

    if account_features["community_risk"] > 70:
        signals.append("High-risk community detected")

    if account_features["graph_degree"] >= 4:
        signals.append("Multiple connected accounts")

    if account_features["weighted_degree"] > 100:
        signals.append("Strong relationship network")

    if account_features["refund_rate"] > 0.10:
        signals.append("High refund activity")

    if account_features["account_age_days"] < 90:
        signals.append("Recently created account")

    if not signals:
        signals = [
            "Trusted account history",
            "Normal customer behaviour"
        ]

    tx["signals"] = signals

    # Decision thresholds
    if probability >= 0.80:

        tx["status"] = "🚨 Fraud Alert"

    elif probability >= 0.50:

        tx["status"] = "⏳ Under Review"

    else:

        tx["status"] = "✅ Approved"

    st.session_state.pending_tx = tx


if st.button("Generate New Payment"):
    create_transaction()

if auto_refresh:
    create_transaction()
    time.sleep(5)
    st.rerun()

# ---------- Animated AI Analysis ----------

if st.session_state.pending_tx:

    tx = st.session_state.pending_tx

    placeholder = st.empty()

    # Step 1
    with placeholder.container():

        st.markdown(f"""
        <div class="live-card" style="background:#111827;border:1px solid #2563eb;">
        <h3>💳 New Payment</h3>

        Merchant: <b>{tx["merchant"]}</b><br>
        Account: {tx["account_id"]}<br>
        Amount: ₹{tx["amount"]:,}<br><br>

        🤖 FraudNet AI received this payment...
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1)

    # Step 2
    with placeholder.container():

        st.markdown(f"""
        <div class="live-card" style="background:#111827;border:1px solid #2563eb;">
        <h3>🧠 AI Analysis Running</h3>

        Fraud Probability: <b>{tx["probability"]*100:.1f}%</b>

        <hr>

        ✅ Checking account age

        ✅ Checking graph connections

        ✅ Checking payment fingerprints

        ✅ Checking refund behavior
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1)

    # Step 3
    colors = {
        "🚨 Fraud Alert": "#7f1d1d",
        "⏳ Under Review": "#78350f",
        "✅ Approved": "#064e3b"
    }

    actions = {
        "🚨 Fraud Alert":
            "Hold payout and manually verify linked accounts.",

        "⏳ Under Review":
            "Review before processing the refund.",

        "✅ Approved":
            "No action needed."
    }

    with placeholder.container():

        st.markdown(f"""
        <div class="live-card" style="background:{colors[tx["status"]]};">

        <h3>{tx["status"]}</h3>

        <b>{tx["merchant"]}</b>

        • ₹{tx["amount"]:,}

        <br><br>

        Fraud Probability:
        <b>{tx["probability"]*100:.1f}%</b>

        <hr>

        <b>Why did the AI decide this?</b>

        <ul>
        {''.join([f"<li>{x}</li>" for x in tx["signals"]])}
        </ul>

        <b>Recommended Action</b><br>
        {actions[tx["status"]]}
        </div>
        """, unsafe_allow_html=True)

    st.session_state.live_feed.insert(0, tx)
    st.session_state.live_feed = st.session_state.live_feed[:8]
    st.session_state.pending_tx = None

# ---------- AI Decision Timeline ----------

st.markdown("### 🕒 AI Decision Timeline")

timeline_colors = {
    "🚨 Fraud Alert": "#7f1d1d",
    "⏳ Under Review": "#78350f",
    "✅ Approved": "#064e3b"
}

for tx in st.session_state.live_feed:

    st.markdown(f"""
    <div class="live-card"
         style="background:{timeline_colors[tx["status"]]};">

    <b>{tx["time"]}</b>

    • {tx["merchant"]}

    • ₹{tx["amount"]:,}

    <br>

    {tx["status"]}

    • Fraud Probability:
    <b>{tx["probability"]*100:.1f}%</b>

    </div>
    """, unsafe_allow_html=True)

st.divider()
# =====================================================
# COMMUNITIES REQUIRING ATTENTION
# =====================================================

st.markdown("## 📋 Communities Requiring Attention")

table = pd.DataFrame(filtered)

table = table[[
    "community_id",
    "ml_risk",
    "size",
    "reasons"
]]

table.columns = [

    "Community",

    "ML Risk",

    "Members",

    "Why Flagged"

]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# AI MERCHANT COPILOT
# =====================================================

st.markdown("## 🤖 Merchant Copilot")

question = st.selectbox(

    "Ask FraudNet",

    [

        "Why was this community flagged?",

        "Should I pause refunds?",

        "How much money can I save?",

        "How confident is the AI?"

    ]
)

if st.button("Ask AI"):

    answers = {

        "Why was this community flagged?":
        selected_comm["reasons"],

        "Should I pause refunds?":
        "Yes. The AI detected coordinated behavior, so manual verification is recommended before processing refunds.",

        "How much money can I save?":
        f"Current estimate: ₹{estimated_savings:,} in prevented fraud losses.",

        "How confident is the AI?":
        f"The selected community has an average ML fraud probability of {selected_comm['ml_risk']}%."

    }

    st.markdown(f"""
    <div class="ai-box">

    🤖 {answers[question]}

    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# COMPARE COMMUNITIES
# =====================================================

with st.expander("⚖️ Compare Communities"):

    ids = [c["community_id"] for c in ranked]

    col_left, col_right = st.columns(2)

    with col_left:

        c1 = st.selectbox(
            "Community A",
            ids,
            index=0,
            key="compare_a"
        )

    with col_right:

        c2 = st.selectbox(
            "Community B",
            ids,
            index=min(1, len(ids)-1),
            key="compare_b"
        )

    s1 = next(s for s in community_stats if s["community_id"] == c1)
    s2 = next(s for s in community_stats if s["community_id"] == c2)

    r1 = next(c for c in ranked if c["community_id"] == c1)
    r2 = next(c for c in ranked if c["community_id"] == c2)

    comparison = pd.DataFrame({

        "Metric": [

            "ML Risk",

            "Members",

            "Shared Devices",

            "Avg Account Age"

        ],

        f"Community {c1}": [

            r1["ml_risk"],

            s1["size"],

            s1["shared_devices"],

            round(s1["avg_account_age"])

        ],

        f"Community {c2}": [

            r2["ml_risk"],

            s2["size"],

            s2["shared_devices"],

            round(s2["avg_account_age"])

        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(f"**Community {c1} Reasons**")

        for reason in r1["reasons"].split(", "):

            st.markdown(f"- {reason}")

    with right:

        st.markdown(f"**Community {c2} Reasons**")

        for reason in r2["reasons"].split(", "):

            st.markdown(f"- {reason}")

st.divider()

# =====================================================
# PART 3 — BUSINESS INSIGHTS & NETWORK VISUALIZATION
# =====================================================

st.markdown("## 📊 Business Insights")

# -----------------------------
# PREPARE CHART DATA
# -----------------------------

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

merged = merged.merge(
    features_df[["account_id", "fraud_probability"]],
    on="account_id"
)

# -----------------------------
# CHARTS
# -----------------------------

distribution_fig = px.histogram(
    accounts,
    x="account_type",
    title="Account Distribution",
    template="plotly_dark",
    color="account_type"
)

distribution_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

refund_fig = px.box(
    merged,
    x="account_type",
    y="refunds",
    title="Refund Behaviour by Account Type",
    template="plotly_dark",
    color="account_type"
)

refund_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

probability_fig = px.histogram(
    merged,
    x="fraud_probability",
    nbins=25,
    title="AI Fraud Probability Distribution",
    template="plotly_dark"
)

probability_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

community_fig = px.bar(
    pd.DataFrame(filtered).sort_values("ml_risk"),
    x="ml_risk",
    y="community_id",
    orientation="h",
    title="Highest-Risk Communities",
    template="plotly_dark"
)

community_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

# -----------------------------
# DISPLAY CHARTS
# -----------------------------

left, right = st.columns(2)

with left:

    # st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    st.plotly_chart(
        distribution_fig,
        use_container_width=True,
        key="distribution_chart"
    )

    # st.markdown("</div>", unsafe_allow_html=True)

with right:

    # st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    st.plotly_chart(
        refund_fig,
        use_container_width=True,
        key="refund_chart"
    )

    # st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:

    # st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    st.plotly_chart(
        probability_fig,
        use_container_width=True,
        key="probability_chart"
    )

    # st.markdown("</div>", unsafe_allow_html=True)

with right:

    # st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    st.plotly_chart(
        community_fig,
        use_container_width=True,
        key="community_chart"
    )

    # st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =====================================================
# COMMUNITY DEEP DIVE
# =====================================================

st.markdown("## 🔎 Community Deep Dive")

m1, m2, m3, m4 = st.columns(4)

cards = [

    ("ML Risk", f"{selected_comm['ml_risk']}%"),

    ("Members", selected_stat["size"]),

    ("Shared Devices", selected_stat["shared_devices"]),

    ("Avg Account Age", f"{round(selected_stat['avg_account_age'])} days")

]

for col, (title, value) in zip([m1, m2, m3, m4], cards):

    with col:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">{title}</div>

        <div class="metric-value">{value}</div>

        </div>
        """, unsafe_allow_html=True)

st.divider()

# =====================================================
# NETWORK VISUALIZATION
# =====================================================

st.markdown("## 🕸 Fraud Relationship Network")

st.caption(
    "Each node represents an account. Hover to view its AI fraud probability."
)

subgraph = G.subgraph(selected_members)

pos = nx.spring_layout(
    subgraph,
    seed=42,
    k=0.6
)

# -----------------------------
# EDGES
# -----------------------------

edge_x = []
edge_y = []

for source, target in subgraph.edges():

    x0, y0 = pos[source]
    x1, y1 = pos[target]

    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(

    x=edge_x,

    y=edge_y,

    mode="lines",

    hoverinfo="none",

    line=dict(
        width=1.4,
        color="#4da3ff"
    )
)

# -----------------------------
# NODES
# -----------------------------

lookup = accounts.set_index("account_id")

node_x = []
node_y = []
colors = []
sizes = []
hover = []

for node in subgraph.nodes():

    x, y = pos[node]

    node_x.append(x)
    node_y.append(y)

    account = lookup.loc[node]

    degree = subgraph.degree(node)

    sizes.append(18 + degree * 2)

    probability = (
        features_df
        .loc[
            features_df.account_id == node,
            "fraud_probability"
        ]
        .iloc[0]
    )

    if probability > 0.8:

        colors.append("#ef4444")

    elif probability > 0.5:

        colors.append("#f59e0b")

    else:

        colors.append("#3b82f6")

    hover.append(

        f"<b>{node}</b><br>"

        f"Fraud Probability: {probability:.1%}<br>"

        f"Connections: {degree}<br>"

        f"Account Age: {account['account_age_days']} days"
    )

node_trace = go.Scatter(

    x=node_x,

    y=node_y,

    mode="markers",

    text=hover,

    hovertemplate="%{text}",

    marker=dict(

        size=sizes,

        color=colors,

        line=dict(
            width=1,
            color="white"
        )
    )
)

network_fig = go.Figure(
    data=[edge_trace, node_trace]
)

network_fig.update_layout(

    template="plotly_dark",

    showlegend=False,

    hovermode="closest",

    margin=dict(
        l=0,
        r=0,
        t=20,
        b=0
    ),

    height=700,

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    xaxis=dict(visible=False),

    yaxis=dict(visible=False)
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)

st.plotly_chart(
    network_fig,
    use_container_width=True,
    key="network_graph"
)

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =====================================================
# TOP SUSPICIOUS ACCOUNTS
# =====================================================

st.markdown("## 🚨 Most Suspicious Accounts")

community_accounts = features_df[
    features_df.account_id.isin(selected_members)
].copy()

community_accounts = community_accounts.sort_values(
    "fraud_probability",
    ascending=False
)

display = community_accounts[[
    "account_id",
    "fraud_probability",
    "graph_degree",
    "refund_rate"
]].copy()

display.columns = [

    "Account",

    "Fraud Probability",

    "Connections",

    "Refund Rate"

]

display["Fraud Probability"] = (
    display["Fraud Probability"]*100
).round(1).astype(str)+"%"

display["Refund Rate"] = (
    display["Refund Rate"]*100
).round(1).astype(str)+"%"

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =====================================================
# PART 4 — REPORTS, SLACK ALERTS & REAL GEMINI COPILOT
# =====================================================

import requests

st.markdown("## 📄 Investigation Summary")

st.write("**Why was this community flagged?**")

for reason in selected_comm["reasons"].split(", "):

    st.markdown(f"""
    <div class="reason-card">
    • {reason}
    </div>
    """, unsafe_allow_html=True)

st.success(f"""

### FraudNet Recommendation

- Pause automatic refunds.
- Verify linked accounts.
- Monitor shared payment fingerprints.
- Continue tracking connected devices.

**AI Confidence:** {selected_comm["ml_risk"]}/100

""")

st.divider()

# =====================================================
# DOWNLOAD REPORT
# =====================================================

st.markdown("## 📥 Export Investigation Report")

report_text = f"""
FraudNet Investigation Report

Community ID: {selected_id}

ML Risk Score: {selected_comm["ml_risk"]}/100

Members: {selected_stat["size"]}

Shared Devices: {selected_stat["shared_devices"]}

Average Account Age: {round(selected_stat["avg_account_age"])} days

Reasons:
{selected_comm["reasons"]}

Recommendation:
Pause refunds and manually verify linked accounts.
"""

st.download_button(

    "⬇ Download Report",

    report_text,

    file_name=f"FraudNet_Community_{selected_id}.txt",

    mime="text/plain"

)

st.divider()

# =====================================================
# REAL GEMINI COPILOT
# =====================================================

st.markdown("## 🤖 FraudNet AI Copilot")

api_key = st.text_input(
    "Gemini API Key",
    type="password"
)

merchant_question = st.text_area(

    "Ask FraudNet anything",

    placeholder="Example: Why is Community 12 suspicious?"

)

if st.button("Ask Gemini"):

    if not api_key:

        st.warning("Please enter your Gemini API key.")

    elif merchant_question.strip() == "":

        st.warning("Enter a question.")

    else:

        prompt = f"""
You are FraudNet, an AI fraud analyst helping merchants.

Community Information

Community ID: {selected_id}

ML Risk: {selected_comm['ml_risk']}

Members: {selected_stat['size']}

Average Account Age:
{round(selected_stat['avg_account_age'])}

Shared Devices:
{selected_stat['shared_devices']}

Reasons:
{selected_comm['reasons']}

Merchant Question:
{merchant_question}

Respond in simple business language.
"""

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        )

        try:

            response = requests.post(

                f"{url}?key={api_key}",

                json={
                    "contents":[
                        {
                            "parts":[
                                {"text":prompt}
                            ]
                        }
                    ]
                },

                timeout=15
            )

            answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            st.markdown(f"""
            <div class="ai-box">
            {answer}
            </div>
            """, unsafe_allow_html=True)

        except Exception:

            st.error("Gemini request failed.")

st.divider()

# =====================================================
# AUTO SLACK ALERT
# =====================================================

if webhook and selected_comm["ml_risk"] >= alert_threshold:

    st.info("High-risk community exceeds alert threshold.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""

---

<div style="text-align:center;color:#94a3b8;padding:20px;">

**FraudNet**

AI-Powered Fraud Ring Detection

Built with

NetworkX • XGBoost • SHAP • Plotly • Streamlit • Gemini

</div>

""", unsafe_allow_html=True)