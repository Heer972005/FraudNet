import networkx as nx
import pandas as pd
import community as community_louvain

def build_graph(accounts_df):

    G = nx.Graph()

    # Add every account as a node
    for _, account in accounts_df.iterrows():

        G.add_node(
            account["account_id"],
            account_type=account["account_type"],
            is_fraud=account["is_fraud_ring"]
        )

    return G


# def connect_by_column(G, accounts_df, column):

#     groups = accounts_df.groupby(column)

#     for _, group in groups:

#         if len(group) < 2:
#             continue

#         accounts = group["account_id"].tolist()

#         for i in range(len(accounts)):
#             for j in range(i+1, len(accounts)):

#                 G.add_edge(
#                     accounts[i],
#                     accounts[j],
#                     reason=column
#                 )
def connect_by_column(G, accounts_df, column, weight):

    groups = accounts_df.groupby(column)

    for _, group in groups:

        if len(group) < 2:
            continue

        accounts = group["account_id"].tolist()

        for i in range(len(accounts)):
            for j in range(i + 1, len(accounts)):

                a = accounts[i]
                b = accounts[j]

                if G.has_edge(a, b):

                    G[a][b]["weight"] += weight

                    G[a][b]["reasons"].append(column)

                else:

                    G.add_edge(
                        a,
                        b,
                        weight=weight,
                        reasons=[column]
                    )



def build_relationship_graph(accounts_df,transactions_df=None,refunds_df=None):
    G = build_graph(accounts_df)

    # connect_by_column(G, accounts_df, "device_id")

    # connect_by_column(G, accounts_df, "ip_address")

    # connect_by_column(
    #     G,
    #     accounts_df,
    #     "payment_method_fingerprint"
    # )
    connect_by_column(
        G,
        accounts_df,
        "device_id",
        weight=3
    )

    connect_by_column(
        G,
        accounts_df,
        "ip_address",
        weight=2
    )

    connect_by_column(
        G,
        accounts_df,
        "payment_method_fingerprint",
        weight=3
    )

    if transactions_df is not None:
        connect_by_merchant_campaign(
        G,
        transactions_df
        )

    if refunds_df is not None:
        connect_by_refund_timing(
        G,
        refunds_df
        )
    return G


def detect_communities(G):

    partition = community_louvain.best_partition(G)

    communities = {}

    for account, community in partition.items():

        communities.setdefault(community, [])

        communities[community].append(account)

    return communities


def calculate_community_stats(G, communities, accounts_df):

    stats = []

    account_lookup = accounts_df.set_index("account_id")

    for community_id, members in communities.items():

        community_accounts = account_lookup.loc[members]
        # subgraph = G.subgraph(members)
        # density = nx.density(nx.subgraph)
        subgraph = G.subgraph(members)
        density = nx.density(subgraph)

        stats.append({
            "community_id": community_id,
            "members": members,
            "size": len(members),
            "density": density,
            "avg_account_age":
                community_accounts["account_age_days"].mean(),

            "shared_devices":
                community_accounts["device_id"].nunique(),

            "shared_payment_methods":
                community_accounts[
                    "payment_method_fingerprint"
                ].nunique(),

            # "fraud_accounts":
            #     community_accounts["is_fraud_ring"].sum()
        })

    return stats

# def score_community(stat):

#     score = 0

#     reasons = []

#     # Large communities
#     if stat["size"] >= 6:
#         score += 20
#         reasons.append("Large cluster")

#     # Shared devices
#     if stat["shared_devices"] <= stat["size"] / 2:
#         score += 25
#         reasons.append("Shared devices")

#     # Shared payment fingerprints
#     if stat["shared_payment_methods"] <= stat["size"] / 2:
#         score += 25
#         reasons.append("Shared payment methods")

#     # Young accounts
#     if stat["avg_account_age"] < 120:
#         score += 15
#         reasons.append("Young accounts")

#     # Known fraud overlap (evaluation only)
#     if stat["fraud_accounts"] >= 2:
#         score += 15
#         reasons.append("Multiple fraud accounts")

#     return min(score,100), reasons

def score_community(stat):

    score = 0
    reasons = []

    # Large community
    if stat["size"] >= 6:
        score += 20
        reasons.append("Large coordinated cluster")

    # Shared devices
    if stat["shared_devices"] <= stat["size"] / 2:
        score += 25
        reasons.append("Heavy device reuse")

    # Shared payment fingerprints
    if stat["shared_payment_methods"] <= stat["size"] / 2:
        score += 25
        reasons.append("Shared payment fingerprints")

    # Young accounts
    if stat["avg_account_age"] < 120:
        score += 10
        reasons.append("Recently created accounts")

    # Graph density
    if stat["density"] > 0.6:
        score += 20
        reasons.append("Dense relationship graph")

    return min(score,100), reasons

def rank_communities(G, communities, accounts_df):

    community_stats = calculate_community_stats(
        G,
        communities,
        accounts_df
    )
    account_lookup = accounts_df.set_index("account_id")
    ranked = []

    for stat in community_stats:

        community_accounts = account_lookup.loc[stat["members"]]
        true_fraud_accounts = community_accounts["is_fraud_ring"].sum()

        score, reasons = score_community(stat)

        ranked.append({
        "community_id": stat["community_id"],
        "risk_score": score,
        "size": stat["size"],
        "true_fraud_accounts": true_fraud_accounts,
        "reasons": ", ".join(reasons)
        })

    return sorted(
        ranked,
        key=lambda x: x["risk_score"],
        reverse=True
    )

def connect_by_merchant_campaign(
    G,
    transactions_df,
    hours=1
):

    transactions = transactions_df.copy()

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    grouped = transactions.groupby("merchant_id")

    for _, group in grouped:

        group = group.sort_values("timestamp")

        rows = group.to_dict("records")

        for i in range(len(rows)):

            for j in range(i+1, len(rows)):

                gap = (
                    rows[j]["timestamp"]
                    - rows[i]["timestamp"]
                ).total_seconds()/3600

                if gap > hours:
                    break

                a = rows[i]["account_id"]
                b = rows[j]["account_id"]

                if a == b:
                    continue

                if G.has_edge(a,b):

                    G[a][b]["weight"] += 2

                    G[a][b]["reasons"].append(
                        "merchant_campaign"
                    )

                else:

                    G.add_edge(
                        a,
                        b,
                        weight=2,
                        reasons=["merchant_campaign"]
                    )


def connect_by_refund_timing(
    G,
    refunds_df,
    hours=24
):

    refunds = refunds_df.copy()

    refunds["refund_timestamp"] = pd.to_datetime(
        refunds["refund_timestamp"]
    )

    refunds = refunds.sort_values("refund_timestamp")

    rows = refunds.to_dict("records")

    for i in range(len(rows)):

        for j in range(i+1, len(rows)):

            gap = (
                rows[j]["refund_timestamp"]
                - rows[i]["refund_timestamp"]
            ).total_seconds()/3600

            if gap > hours:
                break

            if (
                rows[i]["refund_reason"]
                != rows[j]["refund_reason"]
            ):
                continue

            a = rows[i]["account_id"]
            b = rows[j]["account_id"]

            if a == b:
                continue

            if G.has_edge(a,b):

                G[a][b]["weight"] += 3

                G[a][b]["reasons"].append(
                    "refund_timing"
                )

            else:

                G.add_edge(
                    a,
                    b,
                    weight=3,
                    reasons=["refund_timing"]
                )


# if __name__ == "__main__":

#     accounts = pd.read_csv("data/raw/accounts.csv")

#     G = build_relationship_graph(accounts)

#     communities = detect_communities(G)

#     print("Nodes:", G.number_of_nodes())
#     print("Edges:", G.number_of_edges())
#     print("Communities:", len(communities))

#     largest = sorted(
#         communities.values(),
#         key=len,
#         reverse=True
#     )[:5]

#     print("\nLargest Communities:")

#     for i, group in enumerate(largest,1):
#         print(f"{i}. {len(group)} accounts")


if __name__ == "__main__":

    # accounts = pd.read_csv("data/raw/accounts.csv")
    accounts = pd.read_csv("data/raw/accounts.csv")

    transactions = pd.read_csv("data/raw/transactions.csv")

    refunds = pd.read_csv("data/raw/refunds.csv")

    G = build_relationship_graph(accounts,transactions,refunds)

    communities = detect_communities(G)

    ranked = rank_communities(
        G,
        communities,
        accounts
    )

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    top_edges = sorted(
    G.edges(data=True),
    key=lambda x: x[2]["weight"],
    reverse=True
    )[:10]

    print("\nStrongest Relationships\n")

    for a,b,data in top_edges:

        print(
        f"{a} ↔ {b} | "
        f"Weight: {data['weight']} | "
        f"Signals: {', '.join(data['reasons'])}"
        )
    print("Communities:", len(communities))

    print("\nTop 10 Highest-Risk Communities\n")

    for community in ranked[:10]:

        print(
            f"Community {community['community_id']} | "
            f"Risk: {community['risk_score']} | "
            f"Members: {community['size']} | "
            f"Ground Truth Fraud: {community['true_fraud_accounts']}"
        )

        print(f"Reasons: {community['reasons']}\n")