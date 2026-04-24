import hashlib
import networkx as nx
import torch

# Global Graph Storage
transaction_graphs = {}

# Generate Unique Hash for Transaction Groups
def generate_graph_hash(transactions):
    hash_string = "-".join(sorted(transactions))  # Sort for consistency
    return hashlib.sha256(hash_string.encode()).hexdigest()

# Hash Function for Keys
def hash_key(value):
    return hashlib.sha256(value.encode()).hexdigest()

# Add Transaction to Graph
# nodes represent account while edges represent txn.
def add_transaction(txn):
    sender_hash = hash_key(txn["SenderAccount"])
    receiver_hash = hash_key(txn["ReceiverAccount"])

    # Check if sender or receiver is already in a known graph
    related_graphs = [h for h, g in transaction_graphs.items() if sender_hash in g or receiver_hash in g]

    if related_graphs:
        # Merge related graphs into one
        new_graph_hash = generate_graph_hash(related_graphs)
        merged_graph = nx.compose_all([transaction_graphs[h] for h in related_graphs])
        merged_graph.add_edge(sender_hash, receiver_hash, **txn)

        # Remove old graphs and add the merged one
        for h in related_graphs:
            del transaction_graphs[h]
        transaction_graphs[new_graph_hash] = merged_graph
    else:
        # Create a new graph if no related transactions exist
        new_graph = nx.DiGraph()
        new_graph.add_edge(sender_hash, receiver_hash, **txn)   
        transaction_graphs[generate_graph_hash([sender_hash, receiver_hash])] = new_graph

# Detect Laundering Patterns


def detect_pattern(graph):
    """Detect laundering patterns in the transaction graph."""

    # If input is a Torch Geometric graph
    if isinstance(graph, torch.Tensor) or hasattr(graph, "edge_index"):
        # Extract unique node indices
        nodes = torch.unique(graph.edge_index).tolist()
        successors = {node: [] for node in nodes}
        predecessors = {node: [] for node in nodes}

        for i in range(graph.edge_index.shape[1]):  # Process edges
            sender, receiver = graph.edge_index[:, i].tolist()
            successors[sender].append(receiver)
            predecessors[receiver].append(sender)

    # If input is a NetworkX graph
    elif hasattr(graph, "nodes"):
        nodes = list(graph.nodes)
        successors = {node: list(graph.successors(node)) for node in nodes}
        predecessors = {node: list(graph.predecessors(node)) for node in nodes}

    else:
        raise ValueError("Unsupported graph type")

    # Pattern detection logic
    for node in nodes:
        outgoing = successors[node]
        incoming = predecessors[node]

        if len(outgoing) > 5:
            return "Fan-Out"  # One sender, many receivers
        elif len(incoming) > 5:
            return "Fan-In"  # Many senders, one receiver
        elif node in incoming:
            return "Cycle"  # Circular laundering
        elif len(outgoing) > 2 and len(incoming) > 2:
            return "Scatter Gather"  # Money moves across multiple accounts

    return "Normal"



# Store Suspicious AML Clusters
aml_clusters = {}

def flag_suspicious_graph(graph_hash):
    """Mark a graph as an AML cluster if laundering is detected"""
    if graph_hash in transaction_graphs:
        pattern = detect_pattern(transaction_graphs[graph_hash])
        if pattern != "Normal":
            aml_clusters[graph_hash] = transaction_graphs[graph_hash]
            print(f"AML Detected: {pattern} | Cluster ID: {graph_hash}")


