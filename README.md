# FGL-First-Test

A beginner-friendly first implementation of Federated Graph Learning (FGL) using PyTorch Geometric and Flower.

---

## Introduction

This project is my first attempt at building a Federated Graph Learning (FGL) system from scratch.

Instead of reproducing complex research code, the goal is to construct a **minimal, clear, and runnable pipeline** that demonstrates:

- how Graph Neural Networks (GNNs) are trained,
- how Federated Learning (FL) works,
- and how they are combined into FGL.

This repository is designed for learning, experimentation, and future extension.

---

## What is Federated Graph Learning?

Federated Graph Learning (FGL) combines:

- **Graph Neural Networks (GNNs)**  
  Learning from graph-structured data (e.g., citation networks)

- **Federated Learning (FL)**  
  Training models across multiple clients without sharing raw data

In this project:

- each client holds part of the graph data,
- each client trains a local GNN,
- a central server aggregates model parameters using FedAvg.

---

## Project Goals

This repository focuses on:

- building a minimal FGL pipeline
- understanding each component step by step
- keeping code simple and readable
- preparing for future research projects

---

## Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- Flower

---

## Dataset

We use the **Cora** dataset:

- Task: node classification
- Nodes: 2708
- Features: 1433
- Classes: 7

The dataset will be partitioned across multiple clients to simulate a federated environment.

---

## Model

A simple **GCN (Graph Convolutional Network)** is used as the base model.

Focus:

- clarity over complexity
- easy-to-follow forward pass
- suitable for extension

---

## Federated Setting

- Number of clients: 3 (configurable)
- Training: local GNN training per client
- Aggregation: FedAvg
- Multiple communication rounds

---

## Project Structure

```text
fgl-first-test/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── client.py
│   ├── server.py
│   ├── utils.py
│   └── config.py
└── experiments/
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Development Progress

- Project structure initialized
- Environment and dataset verification
- Single-machine GCN training
- Data partitioning into clients
- Federated client implementation
- Server aggregation (FedAvg)
- Federated training experiments
- Evaluation and visualization

## Status

- [x] Load Cora with PyTorch Geometric
- [x] Train single-machine GCN baseline
- [x] Partition training nodes into multiple clients
- [x] Implement local client training
- [x] Implement one-round FedAvg aggregation
- [x] Multi-round federated training
- [ ] Training curve visualization
- [ ] Non-IID client partition
- [ ] Flower-based implementation
