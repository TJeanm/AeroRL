"""Visualize physical body-wrench allocation and its actual PyBullet response."""

import argparse
import json
import math
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import sync
from matplotlib.figure import Figure

from aerorl.control.actuation import cf2x_wrench
from aerorl.control.allocation import CF2XAllocator
from aerorl.utils.simulation import (
    configure_camera,
    follow_camera,
    make_environment,
    rotor_positions_body,
)
from aerorl.utils.telemetry import save_results

EXPERIMENTS = (
    "hover",
    "collective-up",
    "collective-down",
    "roll-positive",
    "roll-negative",
    "pitch-positive",
    "pitch-negative",
    "yaw-positive",
    "yaw-negative",
    "combined",
    "infeasible-torque",
    "infeasible-thrust",
    "negative-thrust",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "motor_allocation"


def make_allocator(env) -> CF2XAllocator:
    return CF2XAllocator(rotor_positions_body(env), env.KF, env.KM, env.MAX_RPM)


def experiment_command(env, name: str) -> tuple[np.ndarray, float]:
    """Physical wrench and pulse duration. Infeasible cases use short probes."""
    weight = env.M * env.G
    command = np.array([weight, 0.0, 0.0, 0.0])
    duration = 0.3
    positions = rotor_positions_body(env)
    torques = weight * np.array(
        [
            np.max(abs(positions[:, 1])) * 0.008,
            np.max(abs(positions[:, 0])) * 0.008,
            env.KM / env.KF * 0.016,
        ]
    )
    if name == "collective-up":
        command[0] *= 1.10
    elif name == "collective-down":
        command[0] *= 0.90
    elif name.startswith(("roll-", "pitch-", "yaw-")):
        axis, sign = name.split("-")
        index = ("roll", "pitch", "yaw").index(axis)
        command[index + 1] = torques[index] * (1 if sign == "positive" else -1)
    elif name == "combined":
        command[0] *= 1.05
        command[1:] = torques * [0.5, -0.4, 0.5]
    elif name == "infeasible-torque":
        command[3] = 0.02  # Too much yaw torque; a single physics tick limits spin-up.
        duration = env.CTRL_TIMESTEP
    elif name == "infeasible-thrust":
        command[0] = 1.2 * make_allocator(env).max_thrust
        duration = 0.12
    elif name == "negative-thrust":
        command[0] = -0.1 * weight
        duration = 0.12
    elif name != "hover":
        raise ValueError(f"Unknown experiment: {name}")
    return command, duration


def run_experiment(env, name: str, *, duration: float = 4.0, camera_distance: float = 0.55):
    if not math.isfinite(duration) or duration < 3:
        raise ValueError("duration must be finite and at least 3 seconds")
    if not math.isfinite(camera_distance) or camera_distance <= 0:
        raise ValueError("camera_distance must be finite and positive")
    obs, _ = env.reset(seed=0)
    positions = rotor_positions_body(env)
    allocator = make_allocator(env)
    command, pulse_duration = experiment_command(env, name)
    allocation = allocator.allocate(command)
    controller = DSLPIDControl(DroneModel.CF2X, g=env.G)
    reference = np.array([0.0, 0.0, 1.0])
    equilibrium = np.array([env.M * env.G, 0.0, 0.0, 0.0])
    client = env.getPyBulletClient()
    if env.GUI:
        configure_camera(env, name, camera_distance)
    rows = []
    start = time.time()
    for step in range(math.ceil(duration * env.CTRL_FREQ)):
        if not p.isConnected(client):
            raise RuntimeError("PyBullet disconnected before the experiment completed")
        t = step / env.CTRL_FREQ
        pid_active = name != "hover" and t >= 1 + pulse_duration
        if pid_active:
            pid_rpm, _, _ = controller.computeControlFromState(
                env.CTRL_TIMESTEP,
                obs[0],
                target_pos=reference,
            )
            # Route PID recovery through the same allocator; this feasible wrench
            # reproduces its RPMs. The PID itself is unchanged.
            requested = cf2x_wrench(pid_rpm, positions, env.KF, env.KM)
            phase = "recovery-pid"
        else:
            requested = command if name == "hover" or t >= 1 else equilibrium
            phase = "wrench-pulse" if t >= 1 and name != "hover" else "hover"
        result = allocator.allocate(requested)
        obs, _, terminated, truncated, _ = env.step(result.rpm.reshape(1, 4))
        state = obs[0].copy()
        if not np.isfinite(state).all() or terminated or truncated:
            raise RuntimeError("Invalid state or unexpected termination in CtrlAviary")
        applied_rpm = state[16:20].copy()
        applied = cf2x_wrench(applied_rpm, positions, env.KF, env.KM)
        rotation = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        rows.append(
            {
                "time_s": (step + 1) / env.CTRL_FREQ,
                "command_time_s": t,
                "phase": phase,
                "pid_active": pid_active,
                "state": state,
                "omega_body": rotation.T @ state[13:16],
                "requested_rpm": result.rpm,
                "applied_rpm": applied_rpm,
                "requested_wrench": requested,
                "applied_wrench": applied,
                "wrench_error": applied - requested,
                "reference_position": reference.copy(),
                "limited": result.limited,
                "collective_clipped": result.collective_clipped,
                "torque_scale": result.torque_scale,
                "saturated_motors": result.saturated_motors,
            }
        )
        if env.GUI:
            if step % 16 == 0:
                follow_camera(env, state[:3].tolist(), camera_distance)
            sync(step, start, env.CTRL_TIMESTEP)
    data = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    distribution = metadata.distribution("gym-pybullet-drones")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    settings = {
        "experiment": name,
        "gym_pybullet_drones_version": distribution.version,
        "upstream_commit": source.get("vcs_info", {}).get("commit_id"),
        "physics": env.PHYSICS.value,
        "pyb_freq_hz": env.PYB_FREQ,
        "ctrl_freq_hz": env.CTRL_FREQ,
        "mass_kg": env.M,
        "gravity_m_s2": env.G,
        "kf_N_per_RPM2": env.KF,
        "km_Nm_per_RPM2": env.KM,
        "inertia_kg_m2": env.J.tolist(),
        "rotor_positions_body_m": positions.tolist(),
        "max_rpm": allocator.max_rpm,
        "allocation_matrix": allocator.matrix.tolist(),
        "pulse_duration_s": pulse_duration,
        "recovery_start_s": 1 + pulse_duration,
        "simulated_duration_s": float(data["time_s"][-1]),
        "pulse_requested_wrench": command.tolist(),
        "pulse_achieved_wrench": allocation.achieved_wrench.tolist(),
        "pulse_wrench_error": allocation.wrench_error.tolist(),
        "pulse_torque_scale": allocation.torque_scale,
        "pulse_limited": allocation.limited,
        "pulse_collective_clipped": allocation.collective_clipped,
        "pulse_saturated_motors": allocation.saturated_motors.tolist(),
        "max_abs_rpy_deg": np.max(
            abs(np.rad2deg(np.unwrap(data["state"][:, 7:10], axis=0))), axis=0
        ).tolist(),
        "max_translation_m": float(np.linalg.norm(data["state"][:, :3] - reference, axis=1).max()),
        "wrench_note": "Body wrench from RPM; excludes gravity, damping and contact. "
        "Requested RPM is the allocator output, not an unconstrained inverse.",
    }
    return data, settings


def plot_allocation_summary(settings: list[dict], path: Path) -> None:
    """Physical requested/achieved wrench for every case, including infeasible ones."""
    figure = Figure(figsize=(14, 12), layout="constrained")
    axes = figure.subplots(5, 1, sharex=True)
    requested = np.array([s["pulse_requested_wrench"] for s in settings])
    achieved = np.array([s["pulse_achieved_wrench"] for s in settings])
    x = np.arange(len(settings))
    for i, label in enumerate(
        ("Thrust (N)", "Roll torque (mN m)", "Pitch torque (mN m)", "Yaw torque (mN m)")
    ):
        scale = 1 if i == 0 else 1000
        axes[i].bar(x - 0.18, scale * requested[:, i], 0.36, label="requested")
        axes[i].bar(x + 0.18, scale * achieved[:, i], 0.36, label="RPM-derived achieved")
        axes[i].set_ylabel(label)
        axes[i].grid(axis="y", alpha=0.25)
        axes[i].legend(fontsize=8)
    colors = ["C3" if s["pulse_limited"] else "C2" for s in settings]
    axes[4].bar(x, [s["pulse_torque_scale"] for s in settings], color=colors)
    axes[4].set_ylabel("Torque scale alpha")
    axes[4].set_xticks(x, [s["experiment"] for s in settings], rotation=40, ha="right")
    axes[4].set_ylim(0, 1.1)
    figure.suptitle(
        "CF2X motor allocation: requested versus achieved\n"
        "Red = limited request; collective takes priority over torque"
    )
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="all")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per case (>=3)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 3:
        parser.error("--duration must be finite and at least 3 seconds")
    output = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    env = make_environment(gui=not args.headless)
    summaries = []
    try:
        for name in names:
            print(f"Running {name}", flush=True)
            data, settings = run_experiment(
                env, name, duration=args.duration, camera_distance=args.camera_distance
            )
            save_results(data, settings, output / name)
            summaries.append(settings)
            print(
                f"  requested: {settings['pulse_requested_wrench']}\n"
                f"  achieved:  {settings['pulse_achieved_wrench']}\n"
                f"  error:     {settings['pulse_wrench_error']}\n"
                f"  limited={settings['pulse_limited']}, "
                f"alpha={settings['pulse_torque_scale']:.6f}, "
                f"motors at limit={settings['pulse_saturated_motors']}",
                flush=True,
            )
        plot_allocation_summary(summaries, output / "allocation_summary.png")
        print(f"Saved {output}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; only completed experiments were saved.")
    finally:
        if p.isConnected(env.getPyBulletClient()):
            env.close()


if __name__ == "__main__":
    main()
