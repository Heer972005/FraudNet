import random
from datetime import datetime

def generate_live_transaction(accounts, selected_members):

    account = accounts.sample(1).iloc[0]

    amount = random.randint(300, 25000)

    transaction = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "account_id": account["account_id"],
        "merchant": random.choice([
            "TechKart",
            "StyleHub",
            "FreshMart",
            "HomeNest",
            "FitFuel"
        ]),
        "amount": amount
    }

    # Fraud decision
    if account["account_id"] in selected_members:

        transaction["status"] = "🚨 Fraud Alert"

        transaction["reason"] = "Linked to high-risk fraud community"

    elif random.random() < 0.08:

        transaction["status"] = "⏳ Under Review"

        transaction["reason"] = "Behavior requires manual verification"

    else:

        transaction["status"] = "✅ Approved"

        transaction["reason"] = "No significant risk detected"

    return transaction