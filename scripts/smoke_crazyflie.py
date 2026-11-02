"""Verify the upstream Crazyflie model and PID controller in PyBullet."""

import argparse
import math
import time

import gymnasium
import numpy as np
import pybullet as p
import stable_baselines3
import torch
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

import aerorl


def run(duration: float, *, gui: bool = True) -> None:
    """Hold one Crazyflie at a fixed position using the existing upstream PID."""
    print(
        f"Loaded {aerorl.__name__}; Gymnasium {gymnasium.__version__}; "
        f"NumPy {np.__version__}; SB3 {stable_baselines3.__version__}; "
        f"PyTorch {torch.__version__}; PyBullet API {p.getAPIVersion()}"
    )
    target = np.array([0.0, 0.0, 0.5])
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=target.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    try:
        controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        obs, _ = env.reset(seed=0)
        client = env.getPyBulletClient()
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=target.tolist(),
                physicsClientId=client,
            )
        start = time.time()
        for step in range(math.ceil(duration * env.CTRL_FREQ)):
            if not p.isConnected(client):
                break
            rpm, _, _ = controller.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=target,
            )
            obs, _, terminated, truncated, _ = env.step(rpm.reshape(1, 4))
            if not np.isfinite(obs).all():
                raise RuntimeError("The simulator returned a non-finite state")
            if terminated or truncated:
                break
            if gui:
                sync(step, start, env.CTRL_TIMESTEP)
        print(f"Smoke test completed; final position (m): {obs[0, :3]}")
    finally:
        if p.isConnected(env.getPyBulletClient()):
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Simulation seconds (default: 5)"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Use PyBullet DIRECT without a window"
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a finite positive number")
    try:
        run(args.duration, gui=not args.headless)
    except KeyboardInterrupt:
        print("Smoke test interrupted; simulator closed.")


if __name__ == "__main__":
    main()
