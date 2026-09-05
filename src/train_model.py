import pandas as pd
import networkx as nx
import numpy as np
from xgboost import XGBClassifier
import shap
from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score
# )
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     precision_score,
#     recall_score,
#     f1_score
# )

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)



import matplotlib.pyplot as plt
from src.graph_detector import (
    build_relationship_graph,
    detect_communities,
    rank_communities,
    calculate_community_stats
)

accounts = pd.read_csv("data/raw/accounts.csv")
transactions = pd.read_csv("data/raw/transactions.csv")
refunds = pd.read_csv("data/raw/refunds.csv")

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

community_lookup = {}

for community in ranked:

    members = next(
        c["members"]
        for c in calculate_community_stats(
            G,
            communities,
            accounts
        )
        if c["community_id"] == community["community_id"]
    )

    for account in members:

        community_lookup[account] = community["risk_score"]



features = []

refund_counts = refunds.groupby("account_id").size().to_dict()

transaction_counts = (
    transactions.groupby("account_id").size().to_dict()
)

for _, account in accounts.iterrows():

    account_id = account["account_id"]

    degree = G.degree(account_id)

    weighted_degree = sum(
        data["weight"]
        for _,_,data in G.edges(
            account_id,
            data=True
        )
    )

    refunds_for_account = refund_counts.get(account_id,0)

    transactions_for_account = transaction_counts.get(account_id,1)

    refund_rate = (
        refunds_for_account
        / transactions_for_account
    )

    features.append({

        "account_age_days":
            account["account_age_days"],

        "community_risk":
            community_lookup.get(account_id,0),

        "graph_degree":
            degree,

        "weighted_degree":
            weighted_degree,

        "refund_rate":
            refund_rate,

        "is_fraud":
            account["is_fraud_ring"]
    })


df = pd.DataFrame(features)
X = df.drop(columns=["is_fraud"])

y = df["is_fraud"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
FRAUD_COST = 5000
FALSE_POSITIVE_COST = 500
model = XGBClassifier(

    n_estimators=200,

    max_depth=5,

    learning_rate=0.05,

    random_state=42,

    eval_metric="logloss"
)

model.fit(X_train,y_train)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# Use a small background sample for faster SHAP computation
background = X_train.sample(
    min(100, len(X_train)),
    random_state=42
)

# Model-agnostic SHAP explainer (works with newer XGBoost versions)
explainer = shap.Explainer(
    model.predict_proba,
    background
)

# Explain only the test set
shap_values = explainer(X_test)

def plot_shap_summary(shap_values, X_test):

    plt.figure(figsize=(8,6))

    shap.summary_plot(
        shap_values[:, :, 1],
        X_test,
        show=False
    )

    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png")
    plt.close()

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:,1]

print(classification_report(y_test,predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test,predictions))

print(
    "\nROC-AUC:",
    roc_auc_score(y_test,probabilities)
)

def simulate_thresholds(y_true, probabilities):

    results = []

    thresholds = np.arange(0.1,0.91,0.1)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y_true,
            predictions
        ).ravel()

        fraud_saved = tp * FRAUD_COST

        fraud_missed = fn * FRAUD_COST

        false_alarm_cost = fp * FALSE_POSITIVE_COST

        net_savings = (
            fraud_saved
            - fraud_missed
            - false_alarm_cost
        )

        results.append({

            "Threshold": threshold,

            "Precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "F1": f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

            "Fraud Caught": tp,

            "Fraud Missed": fn,

            "False Positives": fp,

            "Net Savings": net_savings
        })

    return pd.DataFrame(results)

def plot_roc(y_true, probabilities):

    fpr, tpr, _ = roc_curve(y_true, probabilities)

    plt.figure(figsize=(6,5))

    plt.plot(fpr, tpr, linewidth=2)

    plt.plot([0,1],[0,1],"--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("FraudNet ROC Curve")

    plt.tight_layout()

    plt.savefig("outputs/roc_curve.png")

    plt.close()

def plot_precision_recall(y_true, probabilities):

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities
    )

    plt.figure(figsize=(6,5))

    plt.plot(recall, precision, linewidth=2)

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("FraudNet Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(
        "outputs/precision_recall_curve.png"
    )

    plt.close()

def plot_threshold_savings(results):

    plt.figure(figsize=(7,5))

    plt.plot(
        results["Threshold"],
        results["Net Savings"],
        marker="o",
        linewidth=2
    )

    best = results.loc[
        results["Net Savings"].idxmax()
    ]

    plt.scatter(
        best["Threshold"],
        best["Net Savings"],
        s=120
    )

    plt.text(
        best["Threshold"],
        best["Net Savings"],
        " Best"
    )

    plt.xlabel("Decision Threshold")

    plt.ylabel("Estimated Net Savings (₹)")

    plt.title("Merchant Savings vs Decision Threshold")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("outputs/threshold_savings.png")

    plt.close()

def plot_feature_importance(model, X):

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values()

    plt.figure(figsize=(7,5))

    importance.plot(kind="barh")

    plt.title("FraudNet Feature Importance")

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")

    plt.close()


results = simulate_thresholds(
    y_test,
    probabilities
)

print("\nMerchant Cost Simulation\n")

print(results.to_string(index=False))

best = results.loc[
    results["Net Savings"].idxmax()
]

print("\nRecommended Threshold\n")

print(best)

plot_roc(y_test, probabilities)

plot_precision_recall(
    y_test,
    probabilities
)

plot_threshold_savings(results)

plot_feature_importance(model, X)

print("\nCharts saved inside outputs/")


plot_shap_summary(shap_values, X_test)
highest_risk = probabilities.argmax()

print("\nMost Suspicious Test Account")

print(X_test.iloc[highest_risk])

print(
    "Predicted Fraud Probability:",
    probabilities[highest_risk]
)

def plot_single_explanation(shap_values, index):

    plt.figure(figsize=(8,6))

    shap.plots.waterfall(
        shap_values[index, :, 1],
        show=False
    )

    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall.png")
    plt.close()

plot_single_explanation(
    shap_values,
    highest_risk
)


feature_impacts = pd.DataFrame({

    "Feature": X_test.columns,
    "Contribution": shap_values.values[highest_risk, :, 1]

})

feature_impacts = feature_impacts.sort_values(

    "Contribution",

    ascending=False
)

print("\nTop Feature Contributions")

print(feature_impacts.head())