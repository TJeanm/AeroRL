"""CSV, model metadata and plots shared by the two actuation demonstrations."""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


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
            ax.axvline(settings.get("recovery_start_s", 2), color="gray", alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
    caption = "Open-loop equilibrium throughout"
    if settings["experiment"] == "position":
        caption = "Upstream position PID; target step at 1 s"
    elif settings["experiment"] != "hover":
        caption = (
            f"Open-loop pulse from 1 s; PID recovery from {settings['recovery_start_s']:.3g} s"
        )
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
    for key, names in {
        "wrench_error": ["T_error_N", "tx_error_Nm", "ty_error_Nm", "tz_error_Nm"],
        "saturated_motors": [f"motor{i}_at_limit" for i in range(4)],
        "torque_scale": ["torque_scale"],
        "limited": ["allocation_limited"],
        "collective_clipped": ["collective_clipped"],
    }.items():
        if key in data:
            header.extend(names)
            values = np.asarray(data[key])
            columns.extend(values.T if values.ndim == 2 else [values])
    with (directory / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns, strict=True))
    (directory / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
    plot_telemetry(data, settings, directory / "telemetry.png")
