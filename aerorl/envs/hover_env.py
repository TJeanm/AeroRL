"""Direct thrust/body-torque hover task on the unchanged upstream CF2X simulator."""

import math
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gymnasium import spaces

from aerorl.control.actuation import cf2x_wrench
from aerorl.control.allocation import CF2XAllocator
from aerorl.utils.simulation import configure_camera, follow_camera, rotor_positions_body


class HoverEnv(gym.Env):
    """19 observations, four normalized actions; no PID or rate controller in step().

    Observation: position-target / 1 m, world velocity / 2 m/s, world directions
    of body x and y (six rotation entries), body angular velocity / 5 rad/s,
    previous normalized action. Observations are scaled, never state-clipped.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        control_freq: int = 60,
        episode_seconds: float = 8.0,
        randomization: str = "small",
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if control_freq <= 0 or 240 % control_freq:
            raise ValueError("control_freq must be a positive divisor of 240")
        if not math.isfinite(episode_seconds) or episode_seconds <= 0:
            raise ValueError("episode_seconds must be finite and positive")
        if randomization not in ("none", "easy", "small"):
            raise ValueError("randomization must be none, easy or small")
        self.render_mode = render_mode
        self.control_freq = control_freq
        self.dt = 1 / control_freq
        self.max_steps = math.ceil(episode_seconds * control_freq)
        self.randomization = randomization
        self.target = np.array([0.0, 0.0, 1.0])
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        low = np.r_[np.full(6, -np.inf), np.full(6, -1.0), np.full(3, -np.inf), np.full(4, -1.0)]
        self.observation_space = spaces.Box(
            low=low.astype(np.float32), high=(-low).astype(np.float32)
        )
        self.sim = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.target[None, :],
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=control_freq,
            gui=render_mode == "human",
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self.rotor_positions = rotor_positions_body(self.sim)
        self.allocator = CF2XAllocator(
            self.rotor_positions, self.sim.KF, self.sim.KM, self.sim.MAX_RPM
        )
        self.weight = self.sim.M * self.sim.G
        # Model inertia times conservative angular acceleration limits, in N m.
        self.torque_limits = np.diag(self.sim.J) * np.array([8.0, 8.0, 4.0])
        self.previous_action = np.zeros(4)
        self.state = np.zeros(20)
        self.steps = 0
        self._done = True
        self._recent = deque(maxlen=control_freq)

    def action_to_wrench(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite (4,) array")
        if np.any(abs(action) > 1 + 1e-6):
            raise ValueError("action must lie in [-1,1]")
        action = np.clip(action, -1, 1)  # Numerical boundary tolerance only.
        return np.r_[self.weight * (1 + 0.6 * action[0]), self.torque_limits * action[1:]]

    def _read_state(self) -> np.ndarray:
        client, drone = self.sim.getPyBulletClient(), int(self.sim.getDroneIds()[0])
        pos, quat = p.getBasePositionAndOrientation(drone, physicsClientId=client)
        vel, omega = p.getBaseVelocity(drone, physicsClientId=client)
        return np.r_[pos, quat, p.getEulerFromQuaternion(quat), vel, omega, np.zeros(4)]

    def _observation(self) -> np.ndarray:
        rotation = np.array(p.getMatrixFromQuaternion(self.state[3:7])).reshape(3, 3)
        return np.r_[
            self.state[:3] - self.target,
            self.state[10:13] / 2,
            rotation[:, :2].T.ravel(),
            rotation.T @ self.state[13:16] / 5,
            self.previous_action,
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        amplitude = {
            "none": (0.0, 0.0, 0.0, 0.0),
            "easy": (0.04, 0.03, 3.0, 0.05),
            "small": (0.10, 0.10, 8.0, 0.20),
        }[self.randomization]
        position, velocity, angle_deg, omega = amplitude
        initial = {
            "position": self.target + self.np_random.uniform(-position, position, 3),
            "velocity": self.np_random.uniform(-velocity, velocity, 3),
            "rpy": np.deg2rad(self.np_random.uniform(-angle_deg, angle_deg, 3)),
            "omega_body": self.np_random.uniform(-omega, omega, 3),
        }
        if options:
            for key, value in options.items():
                if key not in initial:
                    raise ValueError(f"Unknown initial-state option: {key}")
                value = np.asarray(value, dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"{key} must be a finite (3,) array")
                initial[key] = value.copy()
        client, drone = self.sim.getPyBulletClient(), int(self.sim.getDroneIds()[0])
        quat = p.getQuaternionFromEuler(initial["rpy"])
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        p.resetBasePositionAndOrientation(drone, initial["position"], quat, physicsClientId=client)
        p.resetBaseVelocity(
            drone, initial["velocity"], rotation @ initial["omega_body"], physicsClientId=client
        )
        self.state = self._read_state()
        self.previous_action = np.zeros(4)
        self.steps, self._done = 0, False
        self._recent.clear()
        self._start_time = time.time()
        if self.render_mode == "human":
            configure_camera(self.sim, "AeroRL hover", 0.7)
        return self._observation(), {"state": self.state.copy(), "initial_state": initial}

    def step(self, action):
        """Policy path: normalized action -> SI wrench -> allocator -> stock physics."""
        return self._advance(self.action_to_wrench(action), np.clip(action, -1, 1))

    def step_wrench(self, wrench):
        """Explicit baseline/diagnostic path; no controller is hidden in this method.

        Used by evaluation's separately instantiated upstream PID at 240 Hz.
        Baseline controls outside RL ranges retain their full effort penalty.
        """
        wrench = np.asarray(wrench, dtype=float)
        action = np.r_[(wrench[0] / self.weight - 1) / 0.6, wrench[1:] / self.torque_limits]
        return self._advance(wrench, action)

    def _advance(self, wrench, action):
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished episode")
        allocation = self.allocator.allocate(wrench)
        raw, *_ = self.sim.step(allocation.rpm[None, :])
        self.steps += 1
        invalid = not np.isfinite(raw).all()
        if not invalid:
            self.state = raw[0].copy()
        # On simulator failure return the last finite state with explicit termination.
        rotation = np.array(p.getMatrixFromQuaternion(self.state[3:7])).reshape(3, 3)
        omega = rotation.T @ self.state[13:16]
        error = self.state[:3] - self.target
        attitude_cost = float((3 - np.trace(rotation)) / 2)
        costs = {
            "position": 4 * float(error @ error),
            "velocity": 0.5 * float(self.state[10:13] @ self.state[10:13]),
            "attitude": 0.5 * max(0.0, attitude_cost),
            "angular_velocity": 0.02 * float(omega @ omega),
            "effort": 0.01 * float(np.mean(np.square(action))),
            "smoothness": 0.01 * float(np.mean(np.square(action - self.previous_action))),
        }
        reward = (1 - sum(costs.values())) * self.dt
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        reason = None
        if invalid:
            reason = "non_finite_state"
        elif self.state[2] < 0.15 or self.state[2] > 2.5:
            reason = "unsafe_altitude"
        elif np.linalg.norm(error) > 1.5:
            reason = "position_limit"
        elif tilt > np.pi / 3:
            reason = "attitude_limit"
        elif p.getContactPoints(
            int(self.sim.getDroneIds()[0]),
            self.sim.PLANE_ID,
            physicsClientId=self.sim.getPyBulletClient(),
        ):
            reason = "ground_contact"
        terminated = reason is not None
        truncated = self.steps >= self.max_steps and not terminated
        if terminated:
            reward -= 2.0
        self._done = terminated or truncated
        self._recent.append(
            [
                error @ error,
                self.state[10:13] @ self.state[10:13],
                omega @ omega,
                tilt,
                abs(self.state[9]),
            ]
        )
        recent = np.asarray(self._recent)
        success = bool(
            truncated
            and len(recent) == self.control_freq
            and np.sqrt(recent[:, 0].mean()) < 0.15
            and np.sqrt(recent[:, 1].mean()) < 0.2
            and np.sqrt(recent[:, 2].mean()) < 0.5
            and recent[:, 3].max() < np.deg2rad(10)
            and recent[:, 4].max() < np.deg2rad(20)
        )
        applied = cf2x_wrench(self.state[16:20], self.rotor_positions, self.sim.KF, self.sim.KM)
        self.previous_action = np.clip(action, -1, 1).copy()
        info = {
            "state": self.state.copy(),
            "omega_body": omega,
            "tilt_rad": tilt,
            "position_error_m": float(np.linalg.norm(error)),
            "is_success": success,
            "termination_reason": reason or ("time_limit" if truncated else None),
            "requested_wrench": np.asarray(wrench).copy(),
            "achieved_wrench": applied,
            "commanded_rpm": allocation.rpm.copy(),
            "wrench_error": applied - wrench,
            "allocation_limited": allocation.limited,
            "torque_scale": allocation.torque_scale,
            "saturated_motors": allocation.saturated_motors,
            "reward_costs": costs,
            "invalid_state": invalid,
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            if self.steps % max(1, self.control_freq // 15) == 0:
                follow_camera(self.sim, self.state[:3].tolist(), 0.7)
            sync(self.steps, self._start_time, self.dt)

    def close(self):
        if p.isConnected(self.sim.getPyBulletClient()):
            self.sim.close()
