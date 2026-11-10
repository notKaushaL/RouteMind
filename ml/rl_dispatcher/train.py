"""
ml/rl_dispatcher/train.py
==========================
Phase 4 — Training Pipeline for Action-Masked PPO Meta-Controller

Trains MaskablePPO over the discrete-event dispatch environment:
  - Policy network: MLP [128, 128] with separate policy and value heads
  - Action masking via ActionMasker wrapper
  - Metric tracking: Episodic reward, SLA breach count, unassigned rate, and route churn
  - Checkpoint saved to: ml/rl_dispatcher/models/maskable_ppo_dispatcher.zip
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

# Silent protobuf / tensorboard compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.modules["tensorboard"] = None

# Ensure RouteMind is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from simulator.environments.dispatch_env import DispatchMetaEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"


def mask_fn(env: DispatchMetaEnv) -> np.ndarray:
    """Extract current boolean action mask from environment."""
    return env.action_masks()


def evaluate_agent(
    model: MaskablePPO,
    eval_episodes: int = 10,
    seed: int = 1234,
) -> Dict[str, float]:
    """
    Evaluate trained MaskablePPO agent over eval_episodes.
    """
    env = DispatchMetaEnv(seed=seed)
    env = ActionMasker(env, mask_fn)

    rewards = []
    delays = []
    unassigned = []
    churns = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += r

        rewards.append(ep_reward)
        delays.append(info.get("max_delay_min", 0.0))
        unassigned.append(info.get("unassigned", 0))
        churns.append(info.get("churn", 0.0))

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_delay_min": float(np.mean(delays)),
        "mean_unassigned": float(np.mean(unassigned)),
        "mean_route_churn": float(np.mean(churns)),
    }


def train_maskable_ppo(
    total_timesteps: int = 20000,
    seed: int = 42,
    lr: float = 3e-4,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    save_path: Optional[Path] = None,
) -> Tuple[MaskablePPO, Dict[str, float]]:
    """
    Train MaskablePPO on DispatchMetaEnv.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = MODELS_DIR / "maskable_ppo_dispatcher.zip"

    log.info("Initializing DispatchMetaEnv with ActionMasker...")
    base_env = DispatchMetaEnv(seed=seed)
    env = ActionMasker(base_env, mask_fn)

    # Multi-layer perceptron with separate policy and value heads
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    log.info("Creating MaskablePPO agent (pi=[128, 128], vf=[128, 128], lr=%.1e)...", lr)
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=256,
        batch_size=64,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )

    log.info("Starting MaskablePPO training for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps)

    # Save model weights
    save_stem = str(save_path).removesuffix(".zip")
    model.save(save_stem)
    log.info("Model saved successfully to %s", save_path)

    # Evaluation
    log.info("Evaluating trained agent over 10 test workdays...")
    metrics = evaluate_agent(model, eval_episodes=10, seed=seed + 999)

    log.info("==================================================")
    log.info(" MaskablePPO Evaluation Results")
    log.info("==================================================")
    log.info(" Mean Episodic Reward: %.2f", metrics["mean_reward"])
    log.info(" Mean Driver Delay:    %.2f min", metrics["mean_delay_min"])
    log.info(" Mean Unassigned:      %.2f stops", metrics["mean_unassigned"])
    log.info(" Mean Route Churn:     %.3f", metrics["mean_route_churn"])

    # Attempt MLflow logging with safe fallback
    try:
        import mlflow
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_experiment("routemind/rl_dispatcher")
        with mlflow.start_run(run_name="maskable_ppo_training"):
            mlflow.log_params({
                "timesteps": total_timesteps,
                "learning_rate": lr,
                "gamma": gamma,
                "ent_coef": ent_coef,
            })
            mlflow.log_metrics(metrics)
            log.info("Logged metrics to MLflow experiment: routemind/rl_dispatcher")
    except Exception as ex:
        log.warning("MLflow logging skipped: %s", ex)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_maskable_ppo(total_timesteps=args.steps, seed=args.seed)
