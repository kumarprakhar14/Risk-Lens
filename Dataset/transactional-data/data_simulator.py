import random
import json
import uuid
from datetime import datetime, timedelta

# Define laundering patterns
PATTERNS = ["Fan-Out", "Fan-In", "Cycle", "Bipartite", "Stack", "Scatter Gather", "Gather Scatter", "Random"]
TRANSACTION_TYPES = ["Credit", "Debit", "Wire", "Card", "Cash"]
TIME_PERIODS = ["Morning", "Afternoon", "Night"]
COUNTRIES = ["USA", "UK", "Canada", "Germany", "India", "China", "UAE", "Australia"]

# Store transaction history
transaction_history = {}

def create_transaction(aml_flag=None):
    """Generate a simulated transaction"""
    sender = f"A{random.randint(1, 1000)}"
    receiver = f"A{random.randint(1, 1000)}"
    amount = random.randint(50, 10000)
    transaction_id = str(uuid.uuid4())

    # Random timestamp (past 6 months)
    timestamp = (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat()

    currency = "USD"
    transaction_type = random.choice(TRANSACTION_TYPES)
    sender_ip = random.choice(COUNTRIES)
    receiver_ip = random.choice(COUNTRIES)

    # Risk Score & AML Flag (Ensuring 50% AML and 50% Normal Transactions)
    if aml_flag is None:
        risk_score = round(random.uniform(0, 1), 2)
        aml_flag = 1 if risk_score > 0.7 else 0
    else:
        risk_score = round(random.uniform(0.8, 1), 2) if aml_flag == 1 else round(random.uniform(0, 0.3), 2)

    # Track transaction history
    sender_history = transaction_history.get(sender, [])
    sender_history.append((amount, timestamp))
    transaction_history[sender] = sender_history[-30:]  # Store last 30 transactions

    # Compute Behavioral & Network Features
    last_24h_tx = [tx for tx in sender_history if datetime.fromisoformat(tx[1]) > datetime.now() - timedelta(days=1)]
    last_7d_tx = [tx for tx in sender_history if datetime.fromisoformat(tx[1]) > datetime.now() - timedelta(days=7)]
    last_30d_tx = sender_history

    daily_tx_count = len(last_24h_tx)
    weekly_tx_count = len(last_7d_tx)
    monthly_tx_count = len(last_30d_tx)
    total_tx_volume = sum(tx[0] for tx in last_30d_tx)

    avg_tx_amount = total_tx_volume / len(last_30d_tx) if last_30d_tx else amount
    tx_amount_deviation = abs(amount - avg_tx_amount)

    new_account_flag = 1 if len(sender_history) < 3 else 0
    dormant_to_active_flag = 1 if random.random() < 0.05 else 0  # 5% chance

    # Transaction Metadata
    weekend_flag = 1 if datetime.fromisoformat(timestamp).weekday() >= 5 else 0
    repeated_amount_flag = 1 if sum(1 for tx in sender_history if tx[0] == amount) > 1 else 0
    time_of_day = random.choice(TIME_PERIODS)

    return {
        "TransactionID": transaction_id,
        "Timestamp": timestamp,
        "SenderAccount": sender,
        "ReceiverAccount": receiver,
        "Amount": amount,
        "Currency": currency,
        "TransactionType": transaction_type,
        "AML_Flag": aml_flag,
        "RiskScore": risk_score,
        "DailyTransactionCount": daily_tx_count,
        "WeeklyTransactionCount": weekly_tx_count,
        "MonthlyTransactionCount": monthly_tx_count,
        "TotalTransactionVolume": total_tx_volume,
        "AverageTransactionAmount": avg_tx_amount,
        "TransactionAmountDeviation": tx_amount_deviation,
        "NewAccountFlag": new_account_flag,
        "DormantToActiveFlag": dormant_to_active_flag,
        "SenderIPLocation": sender_ip,
        "ReceiverIPLocation": receiver_ip,
        "WeekendFlag": weekend_flag,
        "RepeatedAmountFlag": repeated_amount_flag,
        "TimeOfDay": time_of_day
    }

# Generate 50% AML and 50% Normal Transactions
aml_transactions = [create_transaction(aml_flag=1) for _ in range(5000)]  # 50% AML
normal_transactions = [create_transaction(aml_flag=0) for _ in range(5000)]  # 50% Normal
transactions = aml_transactions + normal_transactions
random.shuffle(transactions)

# Save to JSON file
with open("test_transaction_v3.json", "w") as f:
    json.dump(transactions, f, indent=4)

print("Balanced simulated dataset created: simulated_transactions.json")

