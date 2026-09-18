"""
Builds a heterogeneous graph from IEEE-CIS transaction data.
Nodes: transactions, devices, cards, emails, addresses.
Edges: transaction <-> hub entity, for each entity type present.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, LabelEncoder


ENTITY_FIELDS = {
    "device": "DeviceInfo",
    "card": "card1",
    "email": "P_emaildomain",
    "address": "addr1",
}

# Columns that are safe, mostly-complete numeric features for transaction nodes.
# Keep this list small and reliable to start — expand later once the pipeline works.
NUMERIC_FEATURE_COLS = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
]


def load_data(transaction_path: str, identity_path: str) -> pd.DataFrame:
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df


def build_transaction_features(df: pd.DataFrame) -> torch.Tensor:
    """Builds the numeric feature matrix for transaction nodes."""
    feat_df = df[NUMERIC_FEATURE_COLS].copy()
    feat_df = feat_df.fillna(-999)  # simple, explicit missing-value marker
    scaler = StandardScaler()
    feat_array = scaler.fit_transform(feat_df.values)
    return torch.tensor(feat_array, dtype=torch.float)


def build_hetero_graph(df: pd.DataFrame) -> HeteroData:
    data = HeteroData()

    n_transactions = len(df)
    df = df.reset_index(drop=True)
    df["txn_idx"] = df.index  # positional index = node index for transactions

    # --- Transaction nodes ---
    data["transaction"].x = build_transaction_features(df)
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.long)

    # --- Hub nodes + edges, per entity type ---
    for entity_name, col in ENTITY_FIELDS.items():
        values = df[col].astype(str).fillna("missing")

        # Encode unique entity values as hub node indices
        encoder = LabelEncoder()
        hub_idx = encoder.fit_transform(values)
        n_hubs = len(encoder.classes_)

        # Hub nodes get a simple feature: log-count of transactions touching them
        # (a stand-in feature; can be enriched later)
        counts = pd.Series(hub_idx).value_counts().sort_index()
        hub_features = torch.tensor(
            np.log1p(counts.values), dtype=torch.float
        ).unsqueeze(1)
        data[entity_name].x = hub_features

        # Edges: transaction -> hub, and hub -> transaction (bidirectional message passing)
        src = torch.tensor(df["txn_idx"].values, dtype=torch.long)
        dst = torch.tensor(hub_idx, dtype=torch.long)

        data["transaction", f"involves_{entity_name}", entity_name].edge_index = torch.stack([src, dst])
        data[entity_name, f"rev_involves_{entity_name}", "transaction"].edge_index = torch.stack([dst, src])

        print(f"{entity_name}: {n_hubs} unique hub nodes, {len(src)} edges")

    print(f"\nTotal transaction nodes: {n_transactions}")
    return data


if __name__ == "__main__":
    df = load_data(
        "../data/raw/train_transaction.csv",
        "../data/raw/train_identity.csv",
    )
    graph = build_hetero_graph(df)
    print(graph)

    torch.save(graph, "../data/processed/fraud_graph.pt")
    print("\nSaved to data/processed/fraud_graph.pt")
    