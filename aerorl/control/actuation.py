"""Actuation equations for interpreting CF2X commands; no dynamics or controller."""

import numpy as np


def hover_rpm(mass: float, gravity: float, kf: float) -> float:
    """RPM per motor for level, stationary hover: 4 * kf * RPM**2 = mass * g."""
    if not all(np.isfinite(value) and value > 0 for value in (mass, gravity, kf)):
        raise ValueError("mass, gravity and kf must be finite and positive")
    return float(np.sqrt(mass * gravity / (4 * kf)))


def cf2x_wrench(rpm: np.ndarray, rotor_positions: np.ndarray, kf: float, km: float) -> np.ndarray:
    """Return [T, tau_x, tau_y, tau_z] in N and N m, in the body frame.

    Positions are the four rotor COM offsets from the loaded upstream model.
    Use requested RPM to interpret a command, or clipped RPM to interpret what
    PyBullet applied. This calculation does not advance or modify the simulator.
    """
    rpm_squared = np.square(np.asarray(rpm, dtype=float))
    forces = np.zeros((4, 3))
    forces[:, 2] = kf * rpm_squared
    torque = np.cross(rotor_positions, forces).sum(axis=0)
    torque[2] += km * np.dot([-1, 1, -1, 1], rpm_squared)
    return np.array([forces[:, 2].sum(), *torque])
