<div align="center">

# 🕵️ RiskLens: AML Compliance Detection 🚨

**Real-time Anti-Money Laundering Detection using Graph Neural Networks**

Detect complex financial crime patterns with intelligent graph-based learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch)
![GNN](https://img.shields.io/badge/Model-GAT-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

</div>

---

## 🚀 Overview

**RiskLens** is a **Graph Neural Network (GNN)-powered AML detection system** designed for real-time financial transaction monitoring.

Unlike traditional rule-based systems, it captures **hidden relationships and transaction patterns** using graph intelligence.

💡 Built to simulate **bank-grade AML systems** with scalable, real-time detection pipelines.

---

## 🔍 Key Capabilities

✨ Detects complex money laundering patterns:

- 🔁 **Cycle Transactions**
- 🔀 **Fan-In / Fan-Out**
- 🌐 **Scatter-Gather**
- 🔄 **Gather-Scatter**
- 📊 **Anomalous Transaction Clusters**

⚡ Designed for:
- Real-time streaming environments
- High-volume transaction systems
- Advanced fraud detection scenarios

---

## 🧠 How It Works

### 1. 📊 Transaction Graph Construction
- Converts transactions into a **dynamic graph**
- Nodes → Accounts  
- Edges → Transactions  
- Clusters formed based on shared interactions

---

### 2. 🤖 Graph Attention Network (GAT)
- Learns **important transaction relationships using attention**
- Captures **non-linear and hidden fraud patterns**
- Outputs classification:
  - ✅ Normal  
  - 🚨 Suspicious (AML)

---

### 3. ⚡ Real-Time Processing
- Integrated with **Kafka** for streaming pipelines
- Continuously updates graph and predictions
- Enables **low-latency AML detection**

---

## 🏗️ Tech Stack

| Layer | Technology |
|------|------------|
| **ML Framework** | PyTorch |
| **Graph Learning** | PyTorch Geometric |
| **Streaming** | Apache Kafka |
| **Model** | Graph Attention Network (GAT) |

---

## 📈 Performance

- ✅ High accuracy in detecting complex AML patterns  
- 📊 Evaluated using:
  - Confusion Matrix  
  - Precision / Recall  
  - Classification Accuracy  

> Performs effectively on simulated transactional datasets with realistic fraud patterns.

---

## 🎯 Use Cases

- 🏦 Banking & Financial Institutions  
- 💳 Payment Gateways  
- 🕵️ Fraud Detection Systems  
- 📊 Compliance & Risk Analytics  

---

## 📦 Dataset

- **Source:** `Ymak7/transactional-data`
- Simulates real-world financial transaction behavior
- Includes both normal and suspicious patterns

---

## 🔮 Future Improvements

- 🔍 Explainable AI (XAI) for audit transparency  
- ⚙️ Model optimization for large-scale deployment  
- 🌍 Integration with real banking APIs  
- 🧪 Training on real-world AML datasets  

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a feature branch  
3. Submit a PR  

---

## 📬 Contact

For collaborations or queries:

📧 **kumarprakharkp143@gmail.com**

---

<div align="center">

---
license: mit
datasets:
- Ymak7/transactional-data
language:
- en
---

**Built with ❤️ for smarter financial security**

</div>
