"""Paired seeded hover evaluation: deterministic PPO, constant thrust and upstream PID."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from matplotlib.figure import Figure
from stable_baselines3 import PPO

from aerorl.control.actuation import cf2x_wrench
from aerorl.envs.hover_env import HoverEnv

ROOT = Path(__file__).resolve().parents[1] / "results" / "rl_hover"
STATE_COLUMNS = (
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "wx_world_rad_s",
    "wy_world_rad_s",
    "wz_world_rad_s",
    "rpm0",
    "rpm1",
    "rpm2",
    "rpm3",
)


def summarize_metrics(group):
    means = {
        key: float(np.mean([g[key] for g in group]))
        for key in (
            "duration_s",
            "reward",
            "position_rmse_m",
            "final_second_rmse_m",
            "speed_rms_m_s",
            "angular_speed_rms_rad_s",
            "limited_fraction",
            "motor_bound_fraction",
        )
    }
    return {
        "episodes": len(group),
        "success_rate": float(np.mean([g["success"] for g in group])),
        **means,
        "max_position_error_m": max(g["max_position_error_m"] for g in group),
        "mean_episode_max_position_error_m": float(
            np.mean([g["max_position_error_m"] for g in group])
        ),
    }


def evaluate_episode(env, controller, model, seed):
    observation, initial = env.reset(seed=seed)
    pid = DSLPIDControl(DroneModel.CF2X, g=env.sim.G) if controller == "pid" else None
    records = []
    total_reward = 0.0
    while True:
        if controller == "pid":
            rpm, *_ = pid.computeControlFromState(env.dt, env.state, target_pos=env.target)
            wrench = cf2x_wrench(rpm, env.rotor_positions, env.sim.KF, env.sim.KM)
            observation, reward, terminated, truncated, info = env.step_wrench(wrench)
        else:
            action = (
                model.predict(observation, deterministic=True)[0]
                if controller == "ppo"
                else np.zeros(4)
            )
            observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        row = dict(zip(STATE_COLUMNS, info["state"], strict=True))
        row.update(
            time_s=env.steps * env.dt,
            reward=reward,
            position_error_m=info["position_error_m"],
            tilt_rad=info["tilt_rad"],
            allocation_limited=info["allocation_limited"],
            torque_scale=info["torque_scale"],
            motor_at_limit=bool(info["saturated_motors"].any()),
        )
        for key, values in (
            ("omega_body", info["omega_body"]),
            ("requested_wrench", info["requested_wrench"]),
            ("achieved_wrench", info["achieved_wrench"]),
            ("wrench_error", info["wrench_error"]),
            ("commanded_rpm", info["commanded_rpm"]),
        ):
            row.update({f"{key}_{i}": float(value) for i, value in enumerate(values)})
        records.append(row)
        if terminated or truncated:
            break
    error = np.array([r["position_error_m"] for r in records])
    speed = np.array([[r[key] for key in ("vx_m_s", "vy_m_s", "vz_m_s")] for r in records])
    omega = np.array([[r[f"omega_body_{i}"] for i in range(3)] for r in records])
    tail = max(1, env.control_freq)
    metric = {
        "controller": controller,
        "seed": seed,
        "control_hz": env.control_freq,
        "duration_s": env.steps * env.dt,
        "reward": total_reward,
        "success": info["is_success"],
        "termination_reason": info["termination_reason"],
        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "final_second_rmse_m": float(np.sqrt(np.mean(error[-tail:] ** 2))),
        "max_position_error_m": float(error.max()),
        "speed_rms_m_s": float(np.sqrt(np.mean(np.sum(speed**2, axis=1)))),
        "angular_speed_rms_rad_s": float(np.sqrt(np.mean(np.sum(omega**2, axis=1)))),
        "limited_fraction": float(np.mean([r["allocation_limited"] for r in records])),
        "motor_bound_fraction": float(np.mean([r["motor_at_limit"] for r in records])),
    }
    return records, metric, initial["initial_state"]


def plot_episode(records, title, path):
    fig = Figure(figsize=(13, 11), layout="constrained")
    axes = fig.subplots(4, 2).ravel()
    t = np.array([r["time_s"] for r in records])
    for ax, keys, labels, name, factor in (
        (axes[0], ("x_m", "y_m", "z_m"), ("x", "y", "z"), "World position (m)", 1),
        (axes[1], ("vx_m_s", "vy_m_s", "vz_m_s"), ("vx", "vy", "vz"), "Velocity (m/s)", 1),
        (
            axes[2],
            ("roll_rad", "pitch_rad", "yaw_rad"),
            ("roll", "pitch", "yaw"),
            "Attitude (deg)",
            180 / np.pi,
        ),
        (
            axes[3],
            tuple(f"omega_body_{i}" for i in range(3)),
            ("p", "q", "r"),
            "Body rate (rad/s)",
            1,
        ),
        (axes[4], tuple(f"rpm{i}" for i in range(4)), ("0", "1", "2", "3"), "Motor RPM", 1),
        (axes[7], ("position_error_m",), ("error",), "Position error (m)", 1),
    ):
        for key, label in zip(keys, labels, strict=True):
            ax.plot(t, factor * np.array([r[key] for r in records]), label=label)
        ax.set_title(name)
    for i, name in enumerate(("Thrust (N)", "Body torque (mN m)")):
        for j in [0] if i == 0 else [1, 2, 3]:
            factor = 1 if i == 0 else 1000
            axes[5 + i].plot(
                t,
                [factor * r[f"requested_wrench_{j}"] for r in records],
                "--",
                label=f"{j} requested",
            )
            axes[5 + i].plot(
                t, [factor * r[f"achieved_wrench_{j}"] for r in records], label=f"{j} achieved"
            )
        axes[5 + i].set_title(name)
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--controllers", nargs="+", choices=("ppo", "constant", "pid"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000, help="First paired evaluation seed")
    parser.add_argument("--randomization", choices=("none", "easy", "small"), default="small")
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--plot-episodes", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    names = args.controllers or (["ppo", "constant", "pid"] if args.model else ["constant", "pid"])
    if "ppo" in names and args.model is None:
        parser.error("--model is required for PPO evaluation")
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    output = args.output_dir or ROOT / ("evaluation-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu") if "ppo" in names else None
    metrics, initial_states, summary = [], {}, {}
    for name in names:
        env = HoverEnv(
            render_mode="human" if args.gui else None,
            control_freq=240 if name == "pid" else 60,
            episode_seconds=args.episode_seconds,
            randomization=args.randomization,
        )
        try:
            for episode in range(args.episodes):
                seed = args.seed + episode
                records, metric, initial = evaluate_episode(env, name, model, seed)
                metrics.append(metric)
                serialized = {key: value.tolist() for key, value in initial.items()}
                if str(seed) in initial_states:
                    assert initial_states[str(seed)] == serialized, "Baseline initial states differ"
                initial_states[str(seed)] = serialized
                directory = output / name
                directory.mkdir(exist_ok=True)
                with (directory / f"episode_{episode:03}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
                if episode < args.plot_episodes:
                    plot_episode(
                        records, f"{name}; seed {seed}", directory / f"episode_{episode:03}.png"
                    )
            group = [metric for metric in metrics if metric["controller"] == name]
            summary[name] = summarize_metrics(group)
            print(name, json.dumps(summary[name]), flush=True)
        finally:
            env.close()
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "initial_states.json").write_text(json.dumps(initial_states, indent=2) + "\n")
    (output / "config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        )
        + "\n"
    )
    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    for ax, key, label in zip(
        axes,
        ("success_rate", "duration_s", "position_rmse_m"),
        ("Success rate", "Survival (s)", "Episode position RMSE (m)"),
        strict=True,
    ):
        ax.bar(list(summary), [summary[name][key] for name in summary])
        ax.set_title(label)
    fig.savefig(output / "comparison.png", dpi=140)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
