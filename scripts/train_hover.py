"""Train SB3 PPO on direct thrust/body-torque hover; CPU is the default device."""

import argparse
import json
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from aerorl.envs.hover_env import HoverEnv

ROOT = Path(__file__).resolve().parents[1] / "results" / "rl_hover"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Small pipeline check, not a learned model"
    )
    parser.add_argument(
        "--total-timesteps", type=int, help="Additional steps; SB3 rounds to rollout size"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--randomization", choices=("none", "easy", "small"), default="small")
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="SB3 .zip checkpoint; writes into a new run directory"
    )
    args = parser.parse_args()
    timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else (2048 if args.smoke else 300_000)
    )
    if timesteps <= 0 or args.n_envs <= 0:
        parser.error("timesteps and n-envs must be positive")
    directory = args.run_dir or ROOT / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + f"-seed{args.seed}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    n_envs = 1 if args.smoke else args.n_envs

    def factory(index):
        def create():
            return Monitor(
                HoverEnv(randomization=args.randomization, episode_seconds=args.episode_seconds),
                str(directory / f"episodes_{index}.csv"),
                info_keywords=("is_success", "termination_reason"),
            )

        return create

    env = DummyVecEnv([factory(i) for i in range(n_envs)])
    env.seed(args.seed)
    evaluation = Monitor(
        HoverEnv(randomization=args.randomization, episode_seconds=args.episode_seconds)
    )
    evaluation.reset(seed=args.seed + 10_000)
    config = vars(args).copy()
    config.update(
        total_timesteps_requested=timesteps,
        n_envs_actual=n_envs,
        device="cpu",
        observation_dim=19,
        action_dim=4,
        control_hz=60,
        physics_hz=240,
        versions={
            name: metadata.version(name)
            for name in ("gymnasium", "stable-baselines3", "gym-pybullet-drones", "torch")
        },
    )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.monotonic()
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu", seed=args.seed)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                device="cpu",
                seed=args.seed,
                learning_rate=3e-4,
                n_steps=128 if args.smoke else 512,
                batch_size=128 if args.smoke else 256,
                n_epochs=2 if args.smoke else 10,
                gamma=0.995,
                gae_lambda=0.95,
                policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64]), "log_std_init": -1.0},
                verbose=1,
            )
        model.set_logger(configure(str(directory), ["stdout", "csv"]))
        model.save(directory / "starting_model")
        callbacks = [
            CheckpointCallback(
                max(1, (1024 if args.smoke else 100_000) // n_envs),
                str(directory / "checkpoints"),
                name_prefix="hover",
            ),
            EvalCallback(
                evaluation,
                best_model_save_path=str(directory),
                log_path=str(directory),
                eval_freq=max(1, (1024 if args.smoke else 25_000) // n_envs),
                n_eval_episodes=2 if args.smoke else 5,
                deterministic=True,
            ),
        ]
        before = model.num_timesteps
        try:
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                reset_num_timesteps=not bool(args.resume),
            )
        except KeyboardInterrupt:
            model.save(directory / "interrupted_model")
            raise
        model.save(directory / "final_model")
        result = {
            "total_model_timesteps": model.num_timesteps,
            "additional_timesteps": model.num_timesteps - before,
            "wall_seconds": time.monotonic() - start,
        }
        (directory / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {directory}; {result}", flush=True)
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
