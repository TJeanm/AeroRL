"""CF2X body-wrench allocation using squared RPM and explicit motor limits."""

from dataclasses import dataclass

import numpy as np


def allocation_matrix(rotor_positions: np.ndarray, kf: float, km: float) -> np.ndarray:
    """Map RPM squared to [T, tx, ty, tz], in N and N m, motors 0 through 3.

    CF2X rotors are parallel to body +z and yaw reaction signs are [-,+,-,+].
    Geometry must be centered and planar; read it from the loaded upstream model.
    """
    positions = np.asarray(rotor_positions, dtype=float)
    if positions.shape != (4, 3) or not np.isfinite(positions).all():
        raise ValueError("rotor_positions must be a finite (4, 3) array")
    if not all(np.isfinite(value) and value > 0 for value in (kf, km)):
        raise ValueError("kf and km must be finite and positive")
    if not np.allclose(positions[:, 2], 0, atol=1e-9) or not np.allclose(
        positions[:, :2].sum(axis=0), 0, atol=1e-9
    ):
        raise ValueError("CF2X allocation requires centered, planar rotor positions")
    matrix = np.vstack(
        (np.full(4, kf), kf * positions[:, 1], -kf * positions[:, 0], km * np.array([-1, 1, -1, 1]))
    )
    row_norms = np.linalg.norm(matrix, axis=1)
    if np.any(row_norms == 0) or np.linalg.matrix_rank(matrix / row_norms[:, None]) != 4:
        raise ValueError("rotor geometry does not provide four independent wrench axes")
    return matrix


@dataclass
class AllocationResult:
    """Always inspect limited/error; saturated motors alone do not imply infeasibility."""

    rpm: np.ndarray
    squared_rpm: np.ndarray
    requested_wrench: np.ndarray
    achieved_wrench: np.ndarray
    wrench_error: np.ndarray  # Achieved minus requested; units N and N m.
    limited: bool
    collective_clipped: bool
    torque_scale: float
    saturated_motors: np.ndarray  # True at either the lower or upper RPM limit.


class CF2XAllocator:
    """Preserve bounded collective, then uniformly reduce torque to fit all motors.

    This is an algebraic allocator, not an attitude/rate controller. It uses no
    Gymnasium or RL API. Limits are [0, max_rpm] independently for each motor.
    """

    def __init__(self, rotor_positions: np.ndarray, kf: float, km: float, max_rpm: float):
        if not np.isfinite(max_rpm) or max_rpm <= 0:
            raise ValueError("max_rpm must be finite and positive")
        self.matrix = allocation_matrix(rotor_positions, kf, km)
        self.max_rpm = float(max_rpm)
        self.max_squared_rpm = self.max_rpm**2
        self.max_thrust = float(4 * kf * self.max_squared_rpm)

    def allocate(self, wrench: np.ndarray, *, strict: bool = False) -> AllocationResult:
        """Allocate a physical body wrench; strict=True rejects any infeasible request.

        With collective fixed, u = u_collective + alpha * u_torque. Choose the
        largest alpha in [0,1] with every squared RPM in [0,max_rpm**2]. This
        preserves torque direction and ratios, rather than clipping four motors
        independently and accidentally introducing other torque components.
        """
        requested = np.asarray(wrench, dtype=float)
        if requested.shape != (4,) or not np.isfinite(requested).all():
            raise ValueError("wrench must be a finite (4,) array [T, tx, ty, tz]")
        requested = requested.copy()
        thrust = float(np.clip(requested[0], 0, self.max_thrust))
        collective_clipped = bool(thrust != requested[0])
        baseline = np.full(4, thrust / self.max_thrust)  # Squared RPM / max RPM squared.
        delta = np.linalg.solve(self.matrix, np.array([0.0, *requested[1:]]))
        delta /= self.max_squared_rpm
        if not np.isfinite(delta).all():
            raise ValueError("wrench is too large to allocate numerically")
        alpha = 1.0
        for base, change in zip(baseline, delta, strict=True):
            if change > 0:
                alpha = min(alpha, (1 - base) / change)
            elif change < 0:
                alpha = min(alpha, -base / change)
        alpha = float(np.clip(alpha, 0, 1))
        # Only removes floating-point boundary roundoff after analytic allocation.
        fraction = np.clip(baseline + alpha * delta, 0, 1)
        squared_rpm = fraction * self.max_squared_rpm
        achieved = self.matrix @ squared_rpm
        limited = collective_clipped or alpha < 1 - 1e-12
        if strict and limited:
            raise ValueError("infeasible wrench: collective or torque exceeds motor limits")
        return AllocationResult(
            rpm=np.sqrt(squared_rpm),
            squared_rpm=squared_rpm,
            requested_wrench=requested,
            achieved_wrench=achieved,
            wrench_error=achieved - requested,
            limited=limited,
            collective_clipped=collective_clipped,
            torque_scale=alpha,
            saturated_motors=np.isclose(fraction, 0, atol=1e-10, rtol=0)
            | np.isclose(fraction, 1, atol=1e-10, rtol=0),
        )
