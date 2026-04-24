import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ✅ Fix: Import missing F module
import json
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from graph_aml import add_transaction, detect_pattern, transaction_graphs
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight

# Load Simulated Transactions
print("Loading simulated transactions...")
with open("simulated_transactions.json", "r") as f:
    transactions = json.load(f)
print(f"Loaded {len(transactions)} transactions.")

# Define AI Model
class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, heads=3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Apply softmax for classification


# def normalize_feature(x):
#     """Normalize feature vector"""
#     x = np.array(x)
#     return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0) + 1e-8)


# Prepare Graph Data
def normalize_feature(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if np.max(x) - np.min(x) != 0 else x


def prepare_graph():
    print("Preparing graph data...")
    features = []
    edge_list = []
    labels = []
    account_map = {}

    for txn in transactions:
        add_transaction(txn)  # Add transaction to graph

    graph_list = list(transaction_graphs.values())
    print(f"Total transaction graphs created: {len(graph_list)}")

    for i, graph in enumerate(graph_list):
        for node in graph.nodes:
            if node not in account_map:
                account_map[node] = len(account_map)

        for node in graph.nodes:
            raw_feature_vector = [
                len(list(graph.successors(node))),  # Outgoing Connections
                len(list(graph.predecessors(node))),  # Incoming Connections
                1 if detect_pattern(graph) != "Normal" else 0  # AML Label
            ]
            # Normalize features
            feature_vector = [normalize_feature(x) for x in raw_feature_vector]
            features.append(feature_vector)

            labels.append(1 if detect_pattern(graph) != "Normal" else 0)

        for sender, receiver in graph.edges:
            if sender in account_map and receiver in account_map:
                edge_list.append([account_map[sender], account_map[receiver]])

    print("Graph preparation complete.")

    if not features:
        print("No valid features found. Exiting.")
        return None, None

    # 🚨 Debug: Check Label Distribution
    # ✅ Check class balance
    print(f"Label Distribution: {np.bincount(labels)}")

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index), labels


# Train AI Model
def train_gnn():
    print("Starting GNN training...")
    data, labels = prepare_graph()
    if data is None:
        print("Training aborted. No valid data available.")
        return

    model = GAT(num_node_features=3, hidden_dim=16, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    labels_np = np.array(labels).flatten()  # Ensure it's 1D

    # ✅ Ensure both classes exist
    if len(np.unique(labels_np)) < 2:
        print("Warning: Only one class present in dataset! Generating synthetic samples to balance.")

        num_samples = len(labels_np)
        new_class = 1 if np.all(labels_np == 0) else 0  # Add the missing class
        synthetic_samples = np.full((num_samples // 5,), new_class)  # Add 20% of missing class

        labels_np = np.concatenate([labels_np, synthetic_samples])  # Add new samples
        print(f"New Label Distribution: {np.bincount(labels_np)}")  # Debugging

    # Compute class weights after ensuring both classes exist
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=labels_np
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)



    # Apply weighted loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    labels = torch.tensor(labels, dtype=torch.long)
    print("Training started...")

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("GNN Training Complete.")
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved as trained_model.pth")


if __name__ == "__main__":
    train_gnn()
