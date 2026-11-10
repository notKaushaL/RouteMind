"""
ml/rl_dispatcher/agent.py
==========================
Phase 4 — MaskablePPO Meta-Controller Agent

Provides an inference wrapper around the trained MaskablePPO policy:
  - Wraps Gymnasium environment observation and action masks
  - Predicts discrete meta-actions (BUFFER vs CHEAPEST_INSERTION vs STABILITY-RESOLVE)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Silent protobuf / tensorboard compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["tensorboard"] = None

from sb3_contrib import MaskablePPO

log = logging.getLogger(__name__)


class DispatchMetaAgent:
    """
    Inference interface for the trained MaskablePPO Meta-Controller.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, model: Optional[MaskablePPO] = None):
        if model is not None:
            self.model = model
        elif model_path is not None and Path(model_path).exists():
            self.model = MaskablePPO.load(str(model_path))
            log.info("Loaded trained MaskablePPO model from %s", model_path)
        else:
            self.model = None
            log.warning("No MaskablePPO model provided — fallback will be used.")

    def select_action(
        self,
        observation: np.ndarray,
        action_masks: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Select a feasible meta-action given the current observation and boolean action mask.
        """
        if self.model is not None:
            action, _ = self.model.predict(
                observation,
                action_masks=action_masks,
                deterministic=deterministic,
            )
            return int(action)

        # Fallback heuristic if model not trained yet:
        # Prefer cheapest insertion into first feasible vehicle, else BUFFER
        valid_actions = [i for i, m in enumerate(action_masks) if m]
        if not valid_actions:
            return 0

        # Prioritize vehicle insertion (actions 1..V)
        for act in valid_actions:
            if act > 0:
                return act

        return valid_actions[0]
