"""Visual CF2X actuation experiments using upstream PyBullet physics and PID.

Run with --experiment all, or choose one experiment with --help.
See docs/drone_fundamentals.md for equations, motor signs and telemetry units.
"""

import argparse
import csv
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
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from matplotlib.figure import Figure

from aerorl.control.actuation import cf2x_wrench, hover_rpm

EXPERIMENTS = ("hover", "collective-up", "collective-down", "roll", "pitch", "yaw", "position")
PATTERNS = {
    "roll": np.array([-1, -1, 1, 1]),
    "pitch": np.array([-1, 1, 1, -1]),
    "yaw": np.array([-1, 1, -1, 1]),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "drone_fundamentals"


def make_environment(*, gui: bool = False) -> CtrlAviary:
    """Load one stock Crazyflie, initially level and stationary at 1 m."""
    return CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=240,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )


def rotor_positions_body(env: CtrlAviary) -> np.ndarray:
    """Read actual rotor COM offsets; PYB uses these, not nominal L/sqrt(2)."""
    client = env.getPyBulletClient()
    drone = int(env.getDroneIds()[0])
    position, quaternion = p.getBasePositionAndOrientation(drone, physicsClientId=client)
    rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    return np.array(
        [
            rotation.T @ (np.array(p.getLinkState(drone, i, physicsClientId=client)[0]) - position)
            for i in range(4)
        ]
    )


def open_loop_command(experiment: str, t: float, equilibrium_rpm: float) -> tuple[np.ndarray, str]:
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
        if 1.0 <= t < 1.25:
            scale += 0.002 * PATTERNS[experiment]
            phase = "pulse"
        elif 1.25 <= t < 1.5:
            scale -= 0.002 * PATTERNS[experiment]
            phase = "counterpulse"
    return equilibrium_rpm * np.sqrt(scale), phase


def run_experiment(
    env: CtrlAviary, experiment: str, *, duration: float = 6.0
) -> tuple[dict[str, np.ndarray], dict]:
    """Reset, excite, and record one experiment; caller owns environment cleanup."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not math.isfinite(duration) or duration < 3:
        raise ValueError("duration must be finite and at least 3 seconds")
    obs, _ = env.reset(seed=0)
    controller = DSLPIDControl(drone_model=DroneModel.CF2X, g=env.G)
    positions = rotor_positions_body(env)
    equilibrium = hover_rpm(env.M, env.G, env.KF)
    client = env.getPyBulletClient()
    if env.GUI:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 1],
            physicsClientId=client,
        )
        drone = int(env.getDroneIds()[0])
        # Labels and RGB body axes are debug overlays, not custom 3D assets.
        for i, offset in enumerate(positions):
            p.addUserDebugText(
                str(i),
                (offset + [0, 0, 0.03]).tolist(),
                parentObjectUniqueId=drone,
                textSize=1.5,
                textColorRGB=[0, 0, 0],
                physicsClientId=client,
            )
        for axis, color in zip(np.eye(3), np.eye(3), strict=True):
            p.addUserDebugLine(
                [0, 0, 0],
                (0.15 * axis).tolist(),
                color.tolist(),
                parentObjectUniqueId=drone,
                lineWidth=2,
                physicsClientId=client,
            )
        p.addUserDebugText(
            experiment,
            [0, 0, 1.3],
            textSize=1.5,
            textColorRGB=[0, 0, 0],
            physicsClientId=client,
        )
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
        pid_active = experiment == "position" or (experiment != "hover" and t >= 2.0)
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
            rpm, phase = open_loop_command(experiment, t, equilibrium)
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
        "attitude_motor_thrust_fraction": 0.002,
        "timing": "pulse at 1 s; collective reverse at 1.5 s; attitude reverse at 1.25 s; "
        "PID recovery at 2 s; position target step at 1 s",
        "wrench_note": "RPM-derived body wrench, not measured net force/torque; excludes gravity, "
        "contact, damping and optional aerodynamic effects",
    }
    return data, settings


def plot_telemetry(data: dict[str, np.ndarray], settings: dict, path: Path) -> None:
    """Save plots without needing a second GUI or an interactive Matplotlib backend."""
    fig = Figure(figsize=(14, 12), layout="constrained")
    axes = fig.subplots(4, 2).ravel()
    t, command_t, state = data["time_s"], data["command_time_s"], data["state"]
    angles = np.rad2deg(state[:, 7:10]).copy()
    angles[:, 2] = np.rad2deg(np.unwrap(state[:, 9]))
    for ax, values, labels, title in (
        (axes[0], state[:, :3], ("x", "y", "z"), "World position (m)"),
        (axes[1], state[:, 10:13], ("vx", "vy", "vz"), "World velocity (m/s)"),
        (axes[2], angles, ("roll", "pitch", "yaw"), "Attitude (deg; yaw unwrapped)"),
        (axes[3], data["omega_body"], ("p", "q", "r"), "Body angular velocity (rad/s)"),
        (axes[7], state[:, 13:16], ("wx", "wy", "wz"), "World angular velocity (rad/s)"),
    ):
        for i, label in enumerate(labels):
            ax.plot(t, values[:, i], label=label)
        ax.set_title(title)
    for i, label in enumerate(("x", "y", "z")):
        reference = np.where(data["pid_active"], data["reference_position"][:, i], np.nan)
        axes[0].plot(
            command_t, reference, "--", color=f"C{i}", alpha=0.6, label=f"{label} target (PID)"
        )
    for i in range(4):
        axes[4].step(
            command_t,
            data["requested_rpm"][:, i],
            where="post",
            linestyle="--",
            color=f"C{i}",
            alpha=0.6,
            label=f"{i} requested",
        )
        axes[4].step(
            command_t, data["applied_rpm"][:, i], where="post", color=f"C{i}", label=f"{i} applied"
        )
    axes[4].axhline(settings["max_rpm"], color="gray", linestyle=":", label="RPM limit")
    axes[4].set_title("Motor RPM (requested / applied)")
    axes[5].step(
        command_t, data["requested_wrench"][:, 0], where="post", linestyle="--", label="requested"
    )
    axes[5].step(command_t, data["applied_wrench"][:, 0], where="post", label="applied")
    axes[5].axhline(
        settings["mass_kg"] * settings["gravity_m_s2"], color="gray", linestyle=":", label="mg"
    )
    axes[5].set_title("RPM-derived collective thrust (N)")
    for i, label in enumerate(("roll", "pitch", "yaw")):
        axes[6].step(
            command_t,
            1e3 * data["requested_wrench"][:, i + 1],
            where="post",
            linestyle="--",
            color=f"C{i}",
            alpha=0.6,
        )
        axes[6].step(
            command_t,
            1e3 * data["applied_wrench"][:, i + 1],
            where="post",
            color=f"C{i}",
            label=label,
        )
    axes[6].set_title("RPM-derived body torque (mN m; dashed = requested)")
    for ax in axes:
        ax.set_xlabel("Simulation time (s)")
        ax.grid(alpha=0.25)
        if settings["experiment"] != "hover":
            ax.axvline(1, color="gray", alpha=0.3)
        if settings["experiment"] not in ("hover", "position"):
            ax.axvline(2, color="gray", alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
    caption = "Open-loop equilibrium throughout"
    if settings["experiment"] == "position":
        caption = "Upstream position PID; target step at 1 s"
    elif settings["experiment"] != "hover":
        caption = "Open-loop pulses from 1 s; upstream PID recovery from 2 s"
    fig.suptitle(
        f"CF2X: {settings['experiment']} | physics {settings['pyb_freq_hz']} Hz / "
        f"commands {settings['ctrl_freq_hz']} Hz\n{caption}",
        fontsize=14,
    )
    fig.savefig(path, dpi=150)


def save_results(data: dict[str, np.ndarray], settings: dict, directory: Path) -> None:
    """Write CSV in SI units, a parameter/source snapshot, and a readable PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    groups = {
        "state": [
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
            "rpm0_state",
            "rpm1_state",
            "rpm2_state",
            "rpm3_state",
        ],
        "omega_body": [f"{axis}_body_rad_s" for axis in ("p", "q", "r")],
        "requested_rpm": [f"rpm{i}_requested" for i in range(4)],
        "applied_rpm": [f"rpm{i}_applied" for i in range(4)],
        "requested_wrench": [
            "T_requested_N",
            "tx_requested_Nm",
            "ty_requested_Nm",
            "tz_requested_Nm",
        ],
        "applied_wrench": ["T_applied_N", "tx_applied_Nm", "ty_applied_Nm", "tz_applied_Nm"],
        "reference_position": [f"{axis}_reference_m" for axis in ("x", "y", "z")],
    }
    header = ["time_s", "command_time_s", "phase", "pid_active"]
    columns = [data[key] for key in header]
    for key, names in groups.items():
        header.extend(names)
        columns.extend(data[key].T)
    with (directory / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns, strict=True))
    (directory / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
    plot_telemetry(data, settings, directory / "telemetry.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="hover")
    parser.add_argument("--duration", type=float, default=6.0, help="Seconds per experiment (>=3)")
    parser.add_argument("--headless", action="store_true", help="Use PyBullet DIRECT")
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
            data, settings = run_experiment(env, name, duration=args.duration)
            save_results(data, settings, output / name)
            print(f"Saved {output / name}; final position {data['state'][-1, :3]}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; simulator closed. Only completed experiments were saved.")
    finally:
        if p.isConnected(env.getPyBulletClient()):
            env.close()


if __name__ == "__main__":
    main()
