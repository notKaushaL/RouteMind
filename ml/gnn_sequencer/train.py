"""
ml/gnn_sequencer/train.py
==========================
Phase 3 — Training Pipeline for RouteGATv2 Transition Predictor

Trains the GATv2 model using PyTorch, AdamW optimizer, Cosine Annealing LR scheduler,
and early stopping. Evaluates Top-1 accuracy, Top-3 accuracy, and Mean Reciprocal Rank (MRR).

Artifact:
  Saved to ml/gnn_sequencer/models/gatv2_sequencer.pt
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml.gnn_sequencer.dataset import build_or_load_dataset
from ml.gnn_sequencer.model import RouteGATv2, FocalTransitionLoss
from ml.gnn_sequencer.schema import RouteGraphData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def evaluate_graph_transitions(
    model: RouteGATv2,
    graph: RouteGraphData,
    device: torch.device,
) -> Tuple[float, float, float, int]:
    """
    Evaluate Top-1 accuracy, Top-3 accuracy, and MRR for a single dispatch graph.

    Returns:
        (top1_hits, top3_hits, sum_rr, total_evaluated_transitions)
    """
    model.eval()
    data = graph.to_torch(str(device))
    x = data.x if hasattr(data, "x") else data["x"]
    edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else data["edge_attr"]
    y = data.y if hasattr(data, "y") else data["y"]
    num_nodes = graph.num_nodes

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)

    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    top1_hits = 0.0
    top3_hits = 0.0
    sum_rr = 0.0
    valid_transitions = 0

    for u in range(num_nodes):
        mask = (src_idx == u)
        if not mask.any():
            continue

        node_y = y[mask]
        pos_mask = (node_y > 0.5)
        if not pos_mask.any():
            continue

        true_target = dst_idx[mask][torch.argmax(node_y)].item()
        node_logits = logits[mask]
        candidates = dst_idx[mask]

        # Rank candidates by descending logit
        sorted_indices = torch.argsort(node_logits, descending=True)
        ranked_targets = [candidates[idx].item() for idx in sorted_indices]

        if true_target in ranked_targets:
            rank = ranked_targets.index(true_target) + 1
            if rank == 1:
                top1_hits += 1.0
            if rank <= 3:
                top3_hits += 1.0
            sum_rr += 1.0 / rank
            valid_transitions += 1

    return top1_hits, top3_hits, sum_rr, valid_transitions


def evaluate_dataset(
    model: RouteGATv2,
    dataset: List[RouteGraphData],
    criterion: FocalTransitionLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Compute loss, Top-1, Top-3, and MRR over an entire dataset split."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_rr = 0.0
    total_transitions = 0

    with torch.no_grad():
        for graph in dataset:
            data = graph.to_torch(str(device))
            x = data.x if hasattr(data, "x") else data["x"]
            edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else data["edge_attr"]
            y = data.y if hasattr(data, "y") else data["y"]

            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, y, edge_index, graph.num_nodes)
            total_loss += loss.item()

            t1, t3, rr, n_trans = evaluate_graph_transitions(model, graph, device)
            total_top1 += t1
            total_top3 += t3
            total_rr += rr
            total_transitions += n_trans

    n_graphs = max(len(dataset), 1)
    n_trans = max(total_transitions, 1)

    return {
        "val_loss": total_loss / n_graphs,
        "top1_acc": (total_top1 / n_trans) * 100.0,
        "top3_acc": (total_top3 / n_trans) * 100.0,
        "mrr": total_rr / n_trans,
    }


def train_gatv2(
    epochs: int = 15,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    batch_size: int = 16,
    device_str: str = "cpu",
    save_path: Optional[Path] = None,
) -> Tuple[RouteGATv2, Dict[str, float]]:
    """
    Main training function for RouteGATv2.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = MODELS_DIR / "gatv2_sequencer.pt"

    device = torch.device(device_str)
    log.info("Loading / preparing GNN route datasets on device: %s", device)

    train_graphs, val_graphs, test_graphs = build_or_load_dataset(DATA_DIR, n_instances=1500)

    model = RouteGATv2(
        node_in_dim=8,
        edge_in_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=0.10,
    ).to(device)

    criterion = FocalTransitionLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mrr = -1.0
    best_metrics: Dict[str, float] = {}

    log.info("Starting RouteGATv2 training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        # Shuffle training graphs each epoch
        perm = np.random.permutation(len(train_graphs))
        accum_steps = 0
        optimizer.zero_grad()

        for idx in perm:
            graph = train_graphs[idx]
            data = graph.to_torch(str(device))
            x = data.x if hasattr(data, "x") else data["x"]
            edge_index = data.edge_index if hasattr(data, "edge_index") else data["edge_index"]
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else data["edge_attr"]
            y = data.y if hasattr(data, "y") else data["y"]

            logits = model(x, edge_index, edge_attr)
            loss = criterion(logits, y, edge_index, graph.num_nodes)

            # Gradient accumulation across batch_size graphs
            loss = loss / batch_size
            loss.backward()
            train_loss += loss.item() * batch_size
            accum_steps += 1

            if accum_steps % batch_size == 0 or accum_steps == len(train_graphs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_graphs), 1)

        # Validation evaluation (on subset of val to keep training swift)
        val_sub = val_graphs[:60]
        val_metrics = evaluate_dataset(model, val_sub, criterion, device)

        log.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Top-1: %.1f%% | Top-3: %.1f%% | MRR: %.4f",
            epoch, epochs, avg_train_loss, val_metrics["val_loss"],
            val_metrics["top1_acc"], val_metrics["top3_acc"], val_metrics["mrr"]
        )

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "heads": heads,
                    "metrics": val_metrics,
                },
                save_path,
            )
            log.info("  --> Saved new best model (MRR: %.4f) to %s", best_val_mrr, save_path)

    # Final test set evaluation
    log.info("Running final evaluation on test set (%d graphs)...", len(test_graphs))
    checkpoint = torch.load(save_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_dataset(model, test_graphs[:100], criterion, device)

    log.info("==================================================")
    log.info(" GATv2 Final Test Results")
    log.info("==================================================")
    log.info(" Top-1 Accuracy: %.2f%%", test_metrics["top1_acc"])
    log.info(" Top-3 Accuracy: %.2f%%", test_metrics["top3_acc"])
    log.info(" Mean Reciprocal Rank (MRR): %.4f", test_metrics["mrr"])
    log.info(" Model artifact saved: %s", save_path)

    return model, test_metrics


if __name__ == "__main__":
    train_gatv2(epochs=10)
