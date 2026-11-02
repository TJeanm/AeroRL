"""Visual CF2X actuation experiments using upstream PyBullet physics and PID.

Run with --experiment all, or choose one experiment with --help.
See docs/drone_fundamentals.md for equations, motor signs and telemetry units.
"""

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
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import sync

from aerorl.control.actuation import cf2x_wrench, hover_rpm
from aerorl.utils.simulation import (
    configure_camera,
    follow_camera,
    make_environment,
    rotor_positions_body,
)
from aerorl.utils.telemetry import save_results

EXPERIMENTS = ("hover", "collective-up", "collective-down", "roll", "pitch", "yaw", "position")
PATTERNS = {
    "roll": np.array([-1, -1, 1, 1]),
    "pitch": np.array([-1, 1, 1, -1]),
    "yaw": np.array([-1, 1, -1, 1]),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "drone_fundamentals"


def open_loop_command(
    experiment: str,
    t: float,
    equilibrium_rpm: float,
    pulse_magnitude: float = 0.008,
    pulse_duration: float = 0.3,
) -> tuple[np.ndarray, str]:
    """Prescribed motor pulses, expressed as changes in RPM squared (= thrust)."""
    scale = np.ones(4)
    phase = "hover" if experiment == "hover" or t < 1.0 else "coast"
    if experiment.startswith("collective-"):
        direction = 1 if experiment == "collective-up" else -1
        if 1.0 <= t < 1.5:
            scale += direction * 0.10
            phase = "pulse"
        elif 1.5 <= t < 2.0:
            scale -= direction * 0.10
            phase = "counterpulse"
    elif experiment in PATTERNS:
        if 1.0 <= t < 1.0 + pulse_duration:
            scale += pulse_magnitude * PATTERNS[experiment]
            phase = "pulse"
    return equilibrium_rpm * np.sqrt(scale), phase


def run_experiment(
    env: CtrlAviary,
    experiment: str,
    *,
    duration: float = 6.0,
    pulse_magnitude: float = 0.008,
    pulse_duration: float = 0.3,
    camera_distance: float = 0.55,
) -> tuple[dict[str, np.ndarray], dict]:
    """Reset, excite, and record one experiment; caller owns environment cleanup."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not math.isfinite(duration) or duration < 3:
        raise ValueError("duration must be finite and at least 3 seconds")
    if not math.isfinite(pulse_magnitude) or not 0 < pulse_magnitude < 1:
        raise ValueError("pulse_magnitude must be a finite thrust fraction between 0 and 1")
    if not math.isfinite(pulse_duration) or not 0 < pulse_duration < duration - 2:
        raise ValueError("pulse_duration must leave at least 0.9 s for PID recovery")
    if not math.isfinite(camera_distance) or camera_distance <= 0:
        raise ValueError("camera_distance must be finite and positive")
    recovery_start = 1.0 + pulse_duration + 0.1 if experiment in PATTERNS else 2.0
    obs, _ = env.reset(seed=0)
    controller = DSLPIDControl(drone_model=DroneModel.CF2X, g=env.G)
    positions = rotor_positions_body(env)
    equilibrium = hover_rpm(env.M, env.G, env.KF)
    client = env.getPyBulletClient()
    if env.GUI:
        configure_camera(env, experiment, camera_distance)
    records = {
        key: []
        for key in (
            "time_s",
            "command_time_s",
            "state",
            "omega_body",
            "requested_rpm",
            "applied_rpm",
            "requested_wrench",
            "applied_wrench",
            "reference_position",
            "pid_active",
            "phase",
        )
    }
    start = time.time()
    for step in range(math.ceil(duration * env.CTRL_FREQ)):
        if not p.isConnected(client):
            raise RuntimeError("PyBullet disconnected before the experiment completed")
        t = step / env.CTRL_FREQ
        reference = np.array([0.0, 0.0, 1.0])
        pid_active = experiment == "position" or (experiment != "hover" and t >= recovery_start)
        if experiment == "position" and t >= 1.0:
            reference += [0.4, -0.2, 0.2]
        if pid_active:
            rpm, _, _ = controller.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=reference,
            )
            phase = "position-pid" if experiment == "position" else "recovery-pid"
        else:
            rpm, phase = open_loop_command(
                experiment, t, equilibrium, pulse_magnitude, pulse_duration
            )
        obs, _, terminated, truncated, _ = env.step(rpm.reshape(1, 4))
        state = obs[0].copy()
        if not np.isfinite(state).all():
            raise RuntimeError("Simulator returned a non-finite state")
        rotation = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        applied_rpm = state[16:20].copy()
        row = {
            "time_s": (step + 1) / env.CTRL_FREQ,
            "command_time_s": t,
            "state": state,
            "omega_body": rotation.T @ state[13:16],
            "requested_rpm": rpm.copy(),
            "applied_rpm": applied_rpm,
            "requested_wrench": cf2x_wrench(rpm, positions, env.KF, env.KM),
            "applied_wrench": cf2x_wrench(applied_rpm, positions, env.KF, env.KM),
            "reference_position": reference,
            "pid_active": pid_active,
            "phase": phase,
        }
        for key, value in row.items():
            records[key].append(value)
        if terminated or truncated:
            raise RuntimeError("Unexpected termination in upstream CtrlAviary")
        if env.GUI:
            if step % 16 == 0:
                follow_camera(env, state[:3].tolist(), camera_distance)
            sync(step, start, env.CTRL_TIMESTEP)
    data = {key: np.asarray(values) for key, values in records.items()}
    distribution = metadata.distribution("gym-pybullet-drones")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    settings = {
        "experiment": experiment,
        "gym_pybullet_drones_version": distribution.version,
        "upstream_commit": direct_url.get("vcs_info", {}).get("commit_id"),
        "physics": env.PHYSICS.value,
        "drone_model": env.DRONE_MODEL.value,
        "pyb_freq_hz": env.PYB_FREQ,
        "ctrl_freq_hz": env.CTRL_FREQ,
        "requested_duration_s": duration,
        "simulated_duration_s": float(data["time_s"][-1]),
        "mass_kg": env.M,
        "gravity_m_s2": env.G,
        "nominal_arm_m": env.L,
        "inertia_kg_m2": env.J.tolist(),
        "kf_N_per_RPM2": env.KF,
        "km_Nm_per_RPM2": env.KM,
        "rotor_positions_body_m": positions.tolist(),
        "hover_rpm": equilibrium,
        "max_rpm": float(env.MAX_RPM),
        "collective_thrust_fraction": 0.10,
        "attitude_motor_thrust_fraction": pulse_magnitude,
        "pulse_duration_s": pulse_duration,
        "recovery_start_s": recovery_start,
        "camera_distance_m": camera_distance,
        "timing": "pulse at 1 s; attitude pulse then 0.1 s coast then PID; "
        "collective reverse at 1.5 s and PID at 2 s; position target step at 1 s",
        "wrench_note": "RPM-derived body wrench, not measured net force/torque; excludes gravity, "
        "contact, damping and optional aerodynamic effects",
    }
    settings["motion_summary"] = motion_summary(data)
    return data, settings


def motion_summary(data: dict[str, np.ndarray]) -> dict:
    """Maximum absolute Euler excursion and 3D displacement from the initial pose."""
    angles = np.rad2deg(np.unwrap(data["state"][:, 7:10], axis=0))
    return {
        "max_abs_rpy_deg": np.max(np.abs(angles), axis=0).tolist(),
        "max_translation_m": float(np.linalg.norm(data["state"][:, :3] - [0, 0, 1], axis=1).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="hover")
    parser.add_argument("--duration", type=float, default=6.0, help="Seconds per experiment (>=3)")
    parser.add_argument("--headless", action="store_true", help="Use PyBullet DIRECT")
    parser.add_argument(
        "--pulse-magnitude",
        type=float,
        default=0.008,
        help="Attitude per-motor thrust fraction, not RPM fraction (default .008)",
    )
    parser.add_argument(
        "--pulse-duration",
        type=float,
        default=0.3,
        help="Attitude torque pulse seconds, followed by 0.1 s coast and PID",
    )
    parser.add_argument("--camera-distance", type=float, default=0.55, help="Camera distance in m")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 3:
        parser.error("--duration must be finite and at least 3 seconds")
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    # Separate each invocation, including repeated runs of the same experiment.
    output = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    env = make_environment(gui=not args.headless)
    try:
        for name in names:
            print(f"Running {name}: {args.duration:g} s", flush=True)
            data, settings = run_experiment(
                env,
                name,
                duration=args.duration,
                pulse_magnitude=args.pulse_magnitude,
                pulse_duration=args.pulse_duration,
                camera_distance=args.camera_distance,
            )
            save_results(data, settings, output / name)
            print(f"Saved {output / name}; final position {data['state'][-1, :3]}", flush=True)
            print(f"Maximum motion: {settings['motion_summary']}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; simulator closed. Only completed experiments were saved.")
    finally:
        if p.isConnected(env.getPyBulletClient()):
            env.close()


if __name__ == "__main__":
    main()
