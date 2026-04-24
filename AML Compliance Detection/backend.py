import torch
import json
import numpy as np
import time
import networkx as nx
import hashlib
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from torch_geometric.data import Data
from gnn_aml import GAT, prepare_graph
from graph_aml import detect_pattern, add_transaction, hash_key
import logging
import threading
from flask import Flask, jsonify
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockchainGNNFraudDetector:
    def __init__(self, blockchain_node_url, contract_address, contract_abi, model_path="trained_model.pth"):
        """
        Initialize the blockchain-GNN integration for real-time fraud detection
        
        Args:
            blockchain_node_url (str): URL of the blockchain node
            contract_address (str): Address of the smart contract
            contract_abi (list): ABI of the smart contract
            model_path (str): Path to the trained GNN model
        """
        # Initialize blockchain connection
        logger.info("Connecting to blockchain...")
        self.web3 = Web3(Web3.HTTPProvider(blockchain_node_url))
        
        # Add middleware for compatibility with POA chains like BSC, Polygon
        self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to blockchain node")
        
        logger.info(f"Connected to blockchain: {self.web3.is_connected()}")
        
        # Initialize contract
        self.contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=contract_abi
        )
        
        # Load model
        logger.info("Loading GNN model...")
        self.model = GAT(num_node_features=3, hidden_dim=16, output_dim=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Initialize graph structure for transaction analysis
        logger.info("Initializing transaction graphs...")
        self.transaction_graphs = {}  # Dictionary of graph_hash -> nx.DiGraph
        self.graph_index_mapping = {}  # Maps transaction IDs to graph indices
        self.current_graph = None  # Current PyTorch Geometric graph data
        
        # Store transaction history for pattern detection
        self.transaction_history = []
        
        # Transaction buffer for batch processing
        self.transaction_buffer = []
        self.buffer_lock = threading.Lock()
        
        # API server for results
        self.app = Flask(__name__)
        self.detection_results = []
        
        @self.app.route('/api/alerts', methods=['GET'])
        def get_alerts():
            return jsonify(self.detection_results[-100:])  # Return last 100 alerts
    
    def start_api_server(self, port=5000):
        """Start the API server on a separate thread"""
        threading.Thread(target=lambda: self.app.run(host='0.0.0.0', port=port, debug=False), daemon=True).start()
        logger.info(f"API server started on port {port}")
        
    
    # def generate_graph_hash(self, elements):
    #     """Generate a unique hash for a group of elements"""
    #     combined = "".join(sorted(elements))
    #     return hashlib.md5(combined.encode()).hexdigest()
    
    def add_real_time_transaction(self, txn):
        add_transaction(txn)
    
    def convert_nx_to_pytorch_geometric(self):
        """Convert NetworkX graphs to PyTorch Geometric format"""
        # This is a simplified conversion - adjust based on your GAT model's expectations
        # Combine all graphs into one large graph for processing
        combined_graph = nx.compose_all(list(self.transaction_graphs.values()))
        
        # Create node mapping (NetworkX node -> index)
        node_mapping = {node: i for i, node in enumerate(combined_graph.nodes())}
        
        # Extract features 
        num_nodes = len(combined_graph.nodes())

        node_features = torch.zeros((num_nodes, 3), dtype=torch.float)
        
        # Extract edges
        edge_index = []
        for source, target in combined_graph.edges():
            edge_index.append([node_mapping[source], node_mapping[target]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        self.current_graph = Data(x=node_features, edge_index=edge_index)
        
        logger.info(f"Converted NetworkX graphs to PyTorch Geometric format: {num_nodes} nodes, {len(combined_graph.edges())} edges")
    
    def update_graph_with_transaction(self, transaction):
        """
        Update the graph structure with a new blockchain transaction
        
        Args:
            transaction (dict): Transaction details from blockchain
        """
        # Convert blockchain transaction format to your add_transaction format
        txn = {
            "TransactionID": transaction["hash"],
            "SenderAccount": transaction["from"],
            "ReceiverAccount": transaction["to"],
            "Amount": float(transaction["value"]) / 10**18,  # Convert wei to ETH
            "Timestamp": transaction.get("timestamp", time.time())
            # Add any other transaction fields needed for your model
        }
        
        # Add to graph structure
        add_transaction(txn)
    
    def process_transaction(self, transaction):
        """
        Process a blockchain transaction through GNN model
        
        Args:
            transaction (dict): Transaction details
        
        Returns:
            dict: Transaction with fraud prediction results
        """
        # Update graph with this transaction
        self.update_graph_with_transaction(transaction)
        
        # Run prediction only if we have a valid graph
        if self.current_graph is None:
            logger.warning("No graph available for prediction")
            return {
                "TransactionID": transaction["hash"],
                "PredictedLabel": 0,
                "PredictedPattern": "None",
                "RiskScore": 0.0
            }
        
        # Run prediction
        with torch.no_grad():
            output = self.model(self.current_graph)
            probs = torch.softmax(output, dim=1)
            
            # Get prediction for this specific transaction's nodes
            txn_id = transaction["hash"].hex()
            if txn_id in self.graph_index_mapping:
                # Get the node mapping info
                mapping = self.graph_index_mapping[txn_id]
                
                # Get combined risk score from sender and receiver nodes
                # You may need to adjust this logic based on your model's output structure
                node_index = self.map_transaction_to_node(txn_id)
                if node_index is not None:
                    risk_score = probs[node_index, 1].item()
                    prediction = (risk_score > 0.75).long().item()
                else:
                    risk_score = 0.0
                    prediction = 0
            else:
                risk_score = 0.0
                prediction = 0
        
        # Determine AML pattern if suspicious
        if risk_score >= 0.5 and prediction == 1:
            predicted_pattern = detect_pattern(self.current_graph)
        else:
            predicted_pattern = "None"
        
        # Store transaction in history
        result = {
            "TransactionID": transaction["hash"].hex(),
            "PredictedLabel": prediction,
            "PredictedPattern": predicted_pattern,
            "RiskScore": risk_score,
            "Timestamp": time.time()
        }
        
        self.transaction_history.append(result)
        
        # Store the most recent results for API access
        self.detection_results.append(result)
        
        # Log suspicious transactions
        if prediction == 1:
            logger.warning(f"⚠️ Suspicious transaction detected: {transaction['hash'].hex()}, Pattern: {predicted_pattern}")
        
        return result
    
    def map_transaction_to_node(self, transaction_id):
        """
        Map a transaction ID to a node index in the current PyTorch Geometric graph
        
        Args:
            transaction_id (str): Transaction ID
            
        Returns:
            int or None: Node index in the graph, or None if not found
        """
        if transaction_id not in self.graph_index_mapping:
            return None
            
        # Get the sender node hash from the mapping
        sender_hash = self.graph_index_mapping[transaction_id]["sender"]
        
        return 0  
    # def process_fake_transactions(self):
    #     """
    #     Simulates processing fake transactions instead of real blockchain transactions.
    #     """
    #     logger.info("🚀 Running in MOCK mode - Using fake transactions!")

        # Define Fake Transactions
        # fake_transactions = [
        #     {"hash": "0x123", "from": "0xabc", "to": "0xdef", "value": 5 * 10**18},
        #     {"hash": "0x456", "from": "0x111", "to": "0x222", "value": 1000 * 10**18},  # Suspicious?
        # ]

        # for txn in fake_transactions:
        #     logger.info(f"🔍 Processing Fake Transaction: {txn['hash']}")
        #     result = self.process_transaction(txn)
        #     logger.info(f"✅ Prediction Result: {result}")

    
    def start_transaction_listener(self):
        """
        Start listening to real-time blockchain transactions
        """
        logger.info("Starting transaction listener...")
        
        # Set up event filter for your specific transaction events
        # This depends on your contract's events
        try:
            # Get the latest block number
            latest_block = self.web3.eth.block_number
            logger.info(f"Starting from block: {latest_block}")

            while True: 
                try: 
                    # Get current block number
                    curr_block = self.web3.eth.block_number

                    # Process new blocks
                    while latest_block <= curr_block:
                        logger.info(f"Processing block {latest_block}")

                        # Get block data
                        block = self.web3.eth.get_block(latest_block, full_transactions=True)

                        # Process txn in block
                        for transaction in block.transactions:
                            # Check if txm is related to our contract
                            if(transaction['to'] and
                               transaction['to'].lower() == os.getenv('CONTRACT_ADDRESS')):
                                
                                # Add txn. to the buffer
                                self.add_to_buffer(transaction)

                        latest_block += 1

                    # Process buffered txn.
                    self.process_buffer()

                    # Sleep to prevent overwhelming the node
                    time.sleep(1)

                            
                except Exception as e:
                    logger.error(f"Error processing block: {e}")
                    time.sleep(5)   # Wait before retrying

        except Exception as e:
            logger.error(f"Fatal error in transaction listener: {e}")
            raise
            
    def add_to_buffer(self, transaction):
        """Add a transaction to the buffer for batch processing"""
        with self.buffer_lock:
            self.transaction_buffer.append(transaction)
    
    def process_buffer(self):
        """Process all transactions in the buffer"""
        with self.buffer_lock:
            if not self.transaction_buffer:
                return
            
            logger.info(f"Processing {len(self.transaction_buffer)} transactions")
            
            for transaction in self.transaction_buffer:
                self.process_transaction(transaction)
            
            # Clear the buffer
            self.transaction_buffer = []
    
    def start(self, mock_mode=False):
        """Start all components of the detector"""
        """
        Start the fraud detection system. If mock_mode is enabled, it processes fake transactions instead of real blockchain transactions.

        Args:
            mock_mode (bool): Whether to use manually defined transactions for testing.
        """
        # Start API server
        self.start_api_server()

        if mock_mode:
            # 🚀 Use manually defined transactions for testing
            self.process_fake_transactions()
        else:
            # Start real blockchain transaction listener
            threading.Thread(target=self.start_transaction_listener, daemon=True).start()

        logger.info("BlockchainGNNFraudDetector started successfully in MOCK mode" if mock_mode else "LIVE mode")

# Example usage
if __name__ == "__main__":
   # Load contract ABI from file
    with open("Transfer.abi.json", "r") as f:
        contract_abi = json.load(f)
    
    # Initialize detector
    detector = BlockchainGNNFraudDetector(
        blockchain_node_url=os.getenv('blockchain_network_url'),     # blockchain node url
        contract_address=os.getenv("CONTRACT_ADDRESS"),  # contract address
        contract_abi=contract_abi,
        model_path="trained_model.pth"
    )
    
    # Start detector
    detector.start(mock_mode=False)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")