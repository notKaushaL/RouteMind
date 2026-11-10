"""
backend/app/services/engine_registry.py
=======================================
Singleton registry loading trained ML models and inference pipelines once at startup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ml.eta_cqr.model import CQRModel
from ml.gnn_sequencer.model import RouteGATv2
from ml.rl_dispatcher.agent import DispatchMetaAgent

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]


class EngineRegistry:
    """Singleton service maintaining active ML model weights in memory."""

    _instance: Optional["EngineRegistry"] = None

    def __init__(self):
        self.cqr_model: Optional[CQRModel] = None
        self.gatv2_model: Optional[RouteGATv2] = None
        self.rl_agent: Optional[DispatchMetaAgent] = None
        self.is_initialized: bool = False

    @classmethod
    def get_instance(cls) -> "EngineRegistry":
        if cls._instance is None:
            cls._instance = EngineRegistry()
        return cls._instance

    def load_engines(self):
        """Pre-load all models into memory."""
        if self.is_initialized:
            return

        log.info("Initializing RouteMind Engine Registry...")

        # 1. LightGBM CQR Pipeline
        cqr_path = ROOT_DIR / "ml" / "eta_cqr" / "models" / "eta_cqr_pipeline.joblib"
        if cqr_path.exists():
            try:
                self.cqr_model = CQRModel.load(cqr_path)
                log.info("CQR LightGBM pipeline loaded from %s", cqr_path)
            except Exception as e:
                log.warning("Could not load CQR model: %s", e)
        else:
            log.warning("CQR model path not found: %s", cqr_path)

        # 2. GATv2 Sequencer Model
        gat_path = ROOT_DIR / "ml" / "gnn_sequencer" / "models" / "gatv2_sequencer.pt"
        if gat_path.exists():
            try:
                import torch
                model = RouteGATv2(node_dim=8, edge_dim=4, hidden_dim=64, num_heads=4)
                ckpt = torch.load(gat_path, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
                model.eval()
                self.gatv2_model = model
                log.info("GATv2 model loaded from %s", gat_path)
            except Exception as e:
                log.warning("Could not load GATv2 model: %s", e)
        else:
            log.warning("GATv2 model path not found: %s", gat_path)

        # 3. MaskablePPO Agent
        rl_path = ROOT_DIR / "ml" / "rl_dispatcher" / "models" / "maskable_ppo_dispatcher.zip"
        if rl_path.exists():
            try:
                self.rl_agent = DispatchMetaAgent(model_path=rl_path)
                log.info("MaskablePPO agent loaded from %s", rl_path)
            except Exception as e:
                log.warning("Could not load MaskablePPO agent: %s", e)
        else:
            self.rl_agent = DispatchMetaAgent()
            log.warning("MaskablePPO model not found — using rule-based fallback agent.")

        self.is_initialized = True
        log.info("RouteMind Engine Registry initialized successfully.")


registry = EngineRegistry.get_instance()
