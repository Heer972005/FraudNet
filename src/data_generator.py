import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

from src.config import *
import uuid
fake = Faker("en_IN")
random.seed(42)

MERCHANTS = [
    {"id": "M001", "name": "TechKart", "category": "Electronics"},
    {"id": "M002", "name": "StyleHub", "category": "Fashion"},
    {"id": "M003", "name": "FreshMart", "category": "Grocery"},
    {"id": "M004", "name": "HomeNest", "category": "Furniture"},
    {"id": "M005", "name": "FitFuel", "category": "Health"},
]

def random_date_within_last_two_years():
    days = random.randint(0, 730)
    return datetime.now() - timedelta(days=days)


def create_normal_account(index):
    account_age = random.randint(30, 730)
    
    
    return {
        "account_id": f"ACC{index:04d}",
        "customer_name": fake.name(),
        "device_id": f"DEV{random.randint(1000,9999)}",
        "ip_address": fake.ipv4_public(),
        "shipping_address": fake.city(),
        "payment_method_fingerprint": f"PMF{random.randint(1000,9999)}",
        "account_age_days": account_age,
        "refund_probability": round(
            random.uniform(*NORMAL_REFUND_RANGE), 3
        ),
        "account_type": "normal",
        "is_fraud_ring": 0,
        "household_id": None,
        "ring_id": None,
        "scenario_type": "normal",
        "created_date": (
        datetime.now() - timedelta(days=account_age)
        ).strftime("%Y-%m-%d")
    }


def create_shared_household(start_index, household_size):
    """
    Creates a legitimate family/roommate household.
    They share an address and IP, but are NOT fraud.
    """
    
    household_accounts = []

    household_id = f"HOUSE_{uuid.uuid4().hex[:6]}"
    shared_address = fake.city()
    shared_ip = fake.ipv4_public()

    # Most families share Wi-Fi but not always the exact same phone.
    # So we'll create 2–3 devices for the household.
    shared_devices = [
        f"DEV{random.randint(1000,9999)}"
        for _ in range(random.randint(2,3))
    ]

    for i in range(household_size):
        account_age = random.randint(90,730)
        account = {
            "account_id": f"ACC{start_index+i:04d}",
            "customer_name": fake.name(),
            "device_id": random.choice(shared_devices),
            "ip_address": shared_ip,
            "shipping_address": shared_address,
            "payment_method_fingerprint": f"PMF{random.randint(1000,9999)}",
            "account_age_days": account_age,
            "refund_probability": round(
                random.uniform(*NORMAL_REFUND_RANGE),3
            ),
            "account_type": "household",
            "household_id": household_id,
            "is_fraud_ring": 0,
            "ring_id": None,
            "scenario_type": "household",
            "created_date": (
                datetime.now() - timedelta(days=account_age)).strftime("%Y-%m-%d")
        }

        household_accounts.append(account)

    return household_accounts

def create_fraud_ring(start_index, ring_size, ring_type):
    """
    Creates one coordinated fraud ring.
    Different ring types share different signals.
    """
    ring_accounts = []

    ring_id = f"RING_{uuid.uuid4().hex[:6]}"

    shared_device = f"DEV{random.randint(1000,9999)}"
    shared_ip = fake.ipv4_public()
    shared_address = fake.city()
    shared_payment = f"PMF{random.randint(1000,9999)}"

    for i in range(ring_size):
        account_age = random.randint(5,180)
        account = {
            "account_id": f"ACC{start_index+i:04d}",
            "customer_name": fake.name(),
            "account_age_days": account_age,
            "refund_probability": round(
                random.uniform(*FRAUD_REFUND_RANGE),3
            ),
            "account_type": ring_type,
            "ring_id": ring_id,
            "is_fraud_ring": 1,
            "household_id": None,
            "scenario_type": ring_type,
            "created_date": (
                datetime.now() - timedelta(days=account_age)
            ).strftime("%Y-%m-%d")
        }

        # -------- Ring behavior --------

        if ring_type == "obvious":
            account["device_id"] = shared_device
            account["ip_address"] = shared_ip
            account["shipping_address"] = shared_address
            account["payment_method_fingerprint"] = shared_payment

        elif ring_type == "moderate":
            account["device_id"] = shared_device
            account["ip_address"] = fake.ipv4_public()
            account["shipping_address"] = shared_address
            account["payment_method_fingerprint"] = f"PMF{random.randint(1000,9999)}"

        elif ring_type == "sneaky":
            account["device_id"] = f"DEV{random.randint(1000,9999)}"
            account["ip_address"] = fake.ipv4_public()
            account["shipping_address"] = shared_address + f" Sector {random.randint(1,5)}"
            account["payment_method_fingerprint"] = shared_payment
            account["refund_probability"] = round(
                random.uniform(*SNEAKY_REFUND_RANGE),3
            )

        elif ring_type == "expert":
            account["device_id"] = f"DEV{random.randint(1000,9999)}"
            account["ip_address"] = fake.ipv4_public()
            account["shipping_address"] = fake.city()
            account["payment_method_fingerprint"] = shared_payment
            account["refund_probability"] = round(
                random.uniform(*SNEAKY_REFUND_RANGE),3
            )

        ring_accounts.append(account)

    return ring_accounts


def generate_accounts_dataset():
    """
    Creates the complete account dataset containing:
    - Normal customers
    - Legitimate shared households
    - Fraud rings of different difficulty
    """

    accounts = []
    current_index = 1
        # -----------------------------
    # Shared households (legitimate)
    # -----------------------------

    for _ in range(NUM_SHARED_HOUSEHOLDS):

        household_size = random.randint(*HOUSEHOLD_SIZE_RANGE)

        household = create_shared_household(
            current_index,
            household_size
        )

        accounts.extend(household)

        current_index += household_size
    # -----------------------------
    # Fraud rings
    # -----------------------------
    ring_plan = [
        ("obvious", OBVIOUS_RING_COUNT),
            ("moderate", MODERATE_RING_COUNT),
            ("sneaky", SNEAKY_RING_COUNT),
            ("expert", EXPERT_RING_COUNT)
        ]

    for ring_type, count in ring_plan:

        for _ in range(count):

            ring_size = random.randint(
                MIN_RING_SIZE,
                MAX_RING_SIZE
            )

            ring = create_fraud_ring(
                current_index,
                ring_size,
                ring_type
            )

            accounts.extend(ring)

            current_index += ring_size
        # -----------------------------
    # Remaining normal customers
    # -----------------------------

    while len(accounts) < NUM_ACCOUNTS:

        accounts.append(
            create_normal_account(current_index)
        )

        current_index += 1

    df = pd.DataFrame(accounts)

    return df
        
def choose_payment_method(account_type="normal"):

    if account_type in ["obvious","moderate","sneaky","expert"]:
        methods = ["UPI","Card","Wallet"]
        weights = [0.80,0.15,0.05]
    else:
        methods = ["UPI","Card","NetBanking","Wallet"]
        weights = [0.55,0.25,0.10,0.10]

    return random.choices(methods,weights=weights)[0]
def generate_amount(category):
    price_ranges = {
        "Electronics": (800, 5000),
        "Fashion": (400, 2500),
        "Grocery": (150, 1500),
        "Furniture": (2000, 10000),
        "Health": (300, 2500),
    }

    low, high = price_ranges[category]

    return random.randint(low, high)

def generate_campaign_window():
    """
    Creates one coordinated fraud campaign window.
    Every ring gets its own start time.
    """

    start = datetime.now() - timedelta(
        days=random.randint(1, 60),
        hours=random.randint(0, 23)
    )

    return start

def create_transaction(account, transaction_id, timestamp, merchant=None):

    if merchant is None:
        merchant = random.choice(MERCHANTS)

    return {
        "transaction_id": f"TXN{transaction_id:06d}",
        "account_id": account["account_id"],
        "merchant_id": merchant["id"],
        "merchant_name": merchant["name"],
        "merchant_category": merchant["category"],
        "amount": generate_amount(merchant["category"]),
        "payment_method": choose_payment_method(account["account_type"]),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_status": "SUCCESS",
        "hour_of_day": timestamp.hour,
    }


def generate_transactions(accounts_df):

    transactions = []

    txn_counter = 1
    ring_campaigns = {}
    ring_merchants = {}

    for _, account in accounts_df.iterrows():

        account_type = account["account_type"]

        # Different account types shop differently

        if account_type == "normal":
            txn_count = random.randint(2,8)

        elif account_type == "household":
            txn_count = random.randint(3,10)

        elif account_type == "obvious":
            txn_count = random.randint(3,5)

        elif account_type == "moderate":
            txn_count = random.randint(3,6)

        elif account_type == "sneaky":
            txn_count = random.randint(4,6)

        else:   # expert
            txn_count = random.randint(4,7)

        if account_type in ["obvious", "moderate", "sneaky", "expert"]:
            ring_id = account["ring_id"]
            if ring_id not in ring_campaigns:
                ring_campaigns[ring_id] = generate_campaign_window()

            base_time = ring_campaigns[ring_id]

        else:
            base_time = datetime.now() - timedelta(days=random.randint(1,90))

        previous_timestamp = None
        for t in range(txn_count):

            # Fraud timing behavior

            if account_type == "obvious":
                timestamp = base_time + timedelta(
                    minutes=random.randint(0,20)
                )

            elif account_type == "moderate":
                timestamp = base_time + timedelta(
                    hours=random.randint(0,6)
                )

            elif account_type == "sneaky":
                timestamp = base_time + timedelta(
                    days=random.randint(0,5),
                    hours=random.randint(0,12)
                )

            elif account_type == "expert":
                timestamp = base_time + timedelta(
                    days=random.randint(0,30)
                )

            else:
                timestamp = base_time + timedelta(
                    days=t*random.randint(2,15)
                )

            if previous_timestamp is None:
                minutes_since_previous = None
            else:
                minutes_since_previous = int((timestamp - previous_timestamp).total_seconds() / 60)

            previous_timestamp = timestamp
            transaction = create_transaction(
                account,
                txn_counter,
                timestamp
            )
            transaction["minutes_since_previous"] = minutes_since_previous

            # # Fraud merchant behavior

            # if account_type == "obvious":
            #     transaction["merchant_name"] = "TechKart"
            #     transaction["merchant_category"] = "Electronics"
            #     transaction["merchant_id"] = "M001"
            #     transaction["amount"] = random.randint(1800,2200)

            # elif account_type == "moderate":
            #     if random.random() < 0.7:
            #         transaction["merchant_category"] = "Electronics"
            # Fraud merchant behavior

            if account_type in ["obvious", "moderate", "sneaky", "expert"]:
                ring_id = account["ring_id"]
                # First time we see this ring → choose one merchant
                if ring_id not in ring_merchants:
                    ring_merchants[ring_id] = random.choice(MERCHANTS)

                merchant = ring_merchants[ring_id]
                transaction["merchant_id"] = merchant["id"]
                transaction["merchant_name"] = merchant["name"]
                transaction["merchant_category"] = merchant["category"]

                # Keep realistic pricing
                if account_type == "obvious":
                    transaction["amount"] = random.randint(1800,2200)

                elif account_type == "moderate":
                    transaction["amount"] = random.randint(1200,3500)


            transactions.append(transaction)

            txn_counter += 1

    return pd.DataFrame(transactions)

REFUND_REASONS = [
    "Item not received",
    "Damaged item",
    "Wrong product",
    "Duplicate payment",
    "Unauthorized transaction"
]

def create_refund(transaction, account):

    account_type = account["account_type"]

    refund_probability = account["refund_probability"]

    if random.random() > refund_probability:
        return None

    transaction_time = datetime.strptime(
        transaction["timestamp"],
        "%Y-%m-%d %H:%M:%S"
    )

    claim_delay = random.randint(1,14)

    refund_time = transaction_time + timedelta(days=claim_delay)

    # Fraud rings tend to reuse reasons
    if account_type in ["obvious","moderate"]:
        reason = "Item not received"

    elif account_type == "sneaky":
        reason = random.choice([
            "Item not received",
            "Wrong product"
        ])

    else:
        reason = random.choice(REFUND_REASONS)

    return {
        "refund_id": f"REF{transaction['transaction_id'][3:]}",
        "transaction_id": transaction["transaction_id"],
        "account_id": transaction["account_id"],
        "refund_reason": reason,
        "claim_delay_days": claim_delay,
        "refund_timestamp": refund_time.strftime("%Y-%m-%d %H:%M:%S"),
        "refund_status": "Approved",
        "is_fraud_refund": int(account["is_fraud_ring"])
    }


def generate_refunds(transactions_df, accounts_df):

    refunds = []

    account_lookup = accounts_df.set_index("account_id").to_dict("index")

    for _, transaction in transactions_df.iterrows():

        account = account_lookup[transaction["account_id"]]

        refund = create_refund(transaction, account)

        if refund is not None:
            refunds.append(refund)

    return pd.DataFrame(refunds)
# if __name__ == "__main__":
#     sample = create_normal_account(1)

#     print(sample)


# if __name__ == "__main__":

#     household = create_shared_household(1,3)

#     df = pd.DataFrame(household)

#     print(df)

# if __name__ == "__main__":

#     ring = create_fraud_ring(
#         start_index=1,
#         ring_size=5,
#         ring_type="sneaky"
#     )

#     df = pd.DataFrame(ring)

#     print(df)

# if __name__ == "__main__":

#     accounts = generate_accounts_dataset()

#     accounts.to_csv("data/raw/accounts.csv", index=False)

#     print(accounts.head())

#     print("\nAccount Types:")
#     print(accounts["account_type"].value_counts())

#     print(f"\nTotal Accounts: {len(accounts)}")

#     print("\naccounts.csv saved successfully.")


# if __name__ == "__main__":

#     accounts = generate_accounts_dataset()

#     transactions = generate_transactions(accounts)

#     accounts.to_csv(
#         "data/raw/accounts.csv",
#         index=False
#     )

#     transactions.to_csv(
#         "data/raw/transactions.csv",
#         index=False
#     )

#     print("Accounts:", len(accounts))
#     print("Transactions:", len(transactions))

#     print("\nAccount Types:")
#     print(accounts["account_type"].value_counts())

#     print("\nMerchant Categories:")
#     print(transactions["merchant_category"].value_counts())

#     print("\nFiles saved successfully.")


if __name__ == "__main__":

    accounts = generate_accounts_dataset()

    transactions = generate_transactions(accounts)

    refunds = generate_refunds(transactions, accounts)

    accounts.to_csv(
        "data/raw/accounts.csv",
        index=False
    )

    transactions.to_csv(
        "data/raw/transactions.csv",
        index=False
    )

    refunds.to_csv(
        "data/raw/refunds.csv",
        index=False
    )

    print("Accounts:", len(accounts))
    print("Transactions:", len(transactions))
    print("Refunds:", len(refunds))

    print("\nFraud Refunds:")
    print(refunds["is_fraud_refund"].value_counts())

    print("\nFiles saved successfully.")