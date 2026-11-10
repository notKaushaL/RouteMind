"""
ml/gnn_sequencer/model.py
==========================
Phase 3 — RouteGATv2 Transition Predictor

Implements a Graph Attention Network (GATv2) that predicts human driver transition
probabilities P_theta(j | i) from route graph G = (V, E).

Features:
  - Node encoder MLP -> Multi-Head GATv2Conv layers with edge attribute conditioning
  - Pure PyTorch fallback layer (if PyG C++ extensions are unavailable on Windows)
  - Edge scoring head: EdgeMLP([h_i, h_j, e_ij])
  - Row-wise Softmax over outgoing edges -> valid transition probability matrix
  - FocalTransitionLoss: Masked focal cross-entropy addressing sparse edge transitions
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Check PyG GATv2Conv availability
try:
    from torch_geometric.nn import GATv2Conv
    _PYG_GATV2_AVAILABLE = True
except (ImportError, Exception):
    _PYG_GATV2_AVAILABLE = False


class PurePyGATv2Layer(nn.Module):
    """
    Pure PyTorch fallback implementation of GATv2 layer with edge feature integration.
    Does not require any C++ PyG binaries and runs deterministically on Windows.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads

        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)

        self.att = nn.Parameter(torch.empty(1, heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E] (sources, targets)
            edge_attr: [E, edge_dim]
        Returns:
            out: [N, out_dim]
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        h_src = self.lin_src(x[src_idx]).view(-1, self.heads, self.head_dim)
        h_dst = self.lin_dst(x[dst_idx]).view(-1, self.heads, self.head_dim)
        h_edge = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)

        # Dynamic GATv2 score: a^T * LeakyReLU(W_src*h_i + W_dst*h_j + W_edge*e_ij)
        score = (self.att * self.leaky_relu(h_src + h_dst + h_edge)).sum(dim=-1)  # [E, heads]

        # Softmax normalization per destination node
        exp_score = torch.exp(score - score.max(dim=0, keepdim=True)[0])
        # Scatter sum
        sum_exp = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.heads), exp_score)
        alpha = exp_score / (sum_exp[dst_idx] + 1e-8)  # [E, heads]

        # Aggregate weighted source messages
        weighted_msgs = (h_src * alpha.unsqueeze(-1)).view(-1, self.out_dim)  # [E, out_dim]
        out = torch.zeros(N, self.out_dim, device=x.device).scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.out_dim), weighted_msgs)

        return self.norm(out + (x if self.in_dim == self.out_dim else 0.0))


class RouteGATv2(nn.Module):
    """
    Graph Attention Network for Next-Stop Delivery Transition Prediction.
    
    Architecture:
      1. Node Encoder: Linear(8, hidden_dim) -> LayerNorm -> GELU
      2. Multi-Layer GATv2 with Edge Feature Integration (edge_dim=4)
      3. Edge Scoring Head: MLP([h_u, h_v, e_uv]) -> Logit s_uv
    """

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.10,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Node Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # GATv2 Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if _PYG_GATV2_AVAILABLE:
                layer = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_in_dim,
                    concat=True,
                )
            else:
                layer = PurePyGATv2Layer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    heads=heads,
                    edge_dim=edge_in_dim,
                )
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Edge Scoring MLP: [h_src, h_dst, e_attr] -> 1
        edge_mlp_in = 2 * hidden_dim + edge_in_dim
        self.edge_decoder = nn.Sequential(
            nn.Linear(edge_mlp_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing raw transition logits s_ij for each directed edge.

        Args:
            x: [N, 8]
            edge_index: [2, E]
            edge_attr: [E, 4]

        Returns:
            logits: [E] transition logits
        """
        h = self.node_encoder(x)

        for layer, norm in zip(self.layers, self.norms):
            if _PYG_GATV2_AVAILABLE:
                h_new = layer(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = layer(h, edge_index, edge_attr)
            h = norm(h + self.dropout(h_new))

        # Decode transition logits on each directed edge
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = h[src_idx]
        h_dst = h[dst_idx]

        edge_repr = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # [E, 2*H + 4]
        logits = self.edge_decoder(edge_repr).squeeze(-1)          # [E]

        return logits

    def predict_transition_matrix(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute row-normalized transition probability matrix P_theta in R^(N x N).
        P_theta[i, j] is the predicted probability of moving directly from stop i to stop j.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr) / self.temperature

            prob_matrix = torch.zeros((num_nodes, num_nodes), device=x.device, dtype=torch.float32)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # Softmax per source node over valid outgoing edges
            for u in range(num_nodes):
                mask = (src_idx == u)
                if not mask.any():
                    continue
                out_dst = dst_idx[mask]
                out_logits = logits[mask]
                out_probs = F.softmax(out_logits, dim=-1)
                prob_matrix[u, out_dst] = out_probs

            return prob_matrix


class FocalTransitionLoss(nn.Module):
    """
    Focal Cross-Entropy Loss for Next-Stop Transition Modeling.
    
    Addresses the extreme sparsity of positive edge transitions (1 per stop vs N-1 non-transitions)
    by downweighting easy negatives and penalizing hard misclassifications.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            logits: [E] predicted logits
            y: [E] binary labels (1.0 for true transition)
            edge_index: [2, E]
            num_nodes: number of nodes in graph
        """
        src_idx = edge_index[0]
        total_loss = torch.tensor(0.0, device=logits.device)
        valid_nodes = 0

        for u in range(num_nodes):
            mask = (src_idx == u)
            if not mask.any():
                continue

            node_logits = logits[mask]
            node_y = y[mask]

            # Only compute loss if this node had an observed positive transition
            pos_mask = (node_y > 0.5)
            if not pos_mask.any():
                continue

            target_idx = torch.argmax(node_y)
            log_probs = F.log_softmax(node_logits, dim=-1)
            probs = torch.exp(log_probs)

            p_t = probs[target_idx]
            focal_weight = ((1.0 - p_t) ** self.gamma)
            loss_u = -focal_weight * log_probs[target_idx]

            total_loss = total_loss + loss_u
            valid_nodes += 1

        if valid_nodes == 0:
            return total_loss
        return total_loss / valid_nodes
