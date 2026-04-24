import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.data import Data
from gnn_aml import GAT, prepare_graph
from graph_aml import detect_pattern

# Load Model
print("🔍 Loading Trained Model...")
model = GAT(num_node_features=3, hidden_dim=16, output_dim=2)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Load New Test Data
print("Loading New Test Transactions...")
with open("test_transactions_v2.json", "r") as f:
    test_transactions = json.load(f)

# Prepare Graph Data
print("Preparing Test Graph Data...")
test_graph, _ = prepare_graph()

# Run Model Predictions
print("Running Predictions...")
with torch.no_grad():
    output = model(test_graph)
    probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
    predictions = (probs[:, 1] > 0.75).long()  # 1 = AML, 0 = Normal

# Store predictions
test_results = []
y_true = []  # True labels
y_pred = []  # Predicted labels

for txn, prediction in zip(test_transactions, predictions):
    risk_score = txn["RiskScore"]
    true_label = 1 if txn["AML_Flag"] == 1 else 0  # True AML label
    predicted_label = prediction.item()

    # Update labels for confusion matrix
    y_true.append(true_label)
    y_pred.append(predicted_label)

    if risk_score < 0.5:
        predicted_pattern = "None"
    elif predicted_label == 1:
        predicted_pattern = detect_pattern(test_graph)
    else:
        predicted_pattern = "None"

    test_results.append({
        "TransactionID": txn["TransactionID"],
        "TrueLabel": true_label,
        "PredictedLabel": predicted_label,
        "PredictedPattern": predicted_pattern,
        "RiskScore": risk_score
    })

# Save results to file
with open("new_test_results_v2.json", "w") as f:
    json.dump(test_results, f, indent=4)

# **✅ Compute Accuracy Metrics**
print("\n**Final Test Results:**")
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=[
                               "Normal", "AML"], digits=4)

print("\n**Confusion Matrix:**\n", cm)
print("\n**Classification Report:**\n", report)

# **✅ Plot Confusion Matrix**
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Normal", "AML"], yticklabels=["Normal", "AML"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# **✅ Plot Prediction Distribution**
labels, counts = np.unique(y_pred, return_counts=True)
plt.figure(figsize=(6, 5))
plt.bar(["Normal", "AML"], counts, color=["green", "red"])
plt.xlabel("Transaction Classification")
plt.ylabel("Number of Transactions")
plt.title("AML vs. Normal Transactions Detected")
plt.show()

print("Accuracy analysis complete! Check charts & logs.")


# import torch
# import json
# from torch_geometric.data import Data
# from gnn_aml import GAT, prepare_graph
# from graph_aml import detect_pattern

# # Load Model
# print("🔍 Loading Trained Model...")
# model = GAT(num_node_features=3, hidden_dim=16, output_dim=2)
# model.load_state_dict(torch.load("trained_model.pth"))
# model.eval()

# # Load New Test Data
# print("📥 Loading New Test Transactions...")
# with open("test_transactions.json", "r") as f:
#     test_transactions = json.load(f)

# # Prepare Graph Data
# print("🔄 Preparing Test Graph Data...")
# test_graph, _ = prepare_graph()

# # Run Model Predictions
# print("🧠 Running Predictions...")
# with torch.no_grad():
#     output = model(test_graph)
#     probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
#     predictions = (probs[:, 1] > 0.75).long()  # 1 = AML, 0 = Normal

# # Store predictions
# test_results = []
# aml_count = 0
# normal_count = 0

# for txn, prediction in zip(test_transactions, predictions):
#     risk_score = txn["RiskScore"]
#     predicted_label = prediction.item()

#     if risk_score < 0.5:
#         predicted_pattern = "None"  # ✅ Mark as safe
#         normal_count += 1  # ✅ Count normal transactions
#     elif predicted_label == 1:
#         predicted_pattern = detect_pattern(
#             test_graph)  # ✅ Detect actual pattern
#         aml_count += 1  # ✅ Count AML transactions
#     else:
#         predicted_pattern = "None"
#         normal_count += 1  # ✅ Count normal transactions

#     test_results.append({
#         "TransactionID": txn["TransactionID"],
#         "PredictedPattern": predicted_pattern,
#         "RiskScore": risk_score
#     })

# # **✅ Move logging here, after results are fully analyzed**
# print("\n📊 **Final Test Results:**")
# print(f"🔴 AML Detected: {aml_count}")
# print(f"🟢 Normal Transactions: {normal_count}")

# # Save results to file
# with open("new_test_results_v2.json", "w") as f:
#     json.dump(test_results, f, indent=4)

# print("✅ Test results saved to `new_test_results_v2.json`")
