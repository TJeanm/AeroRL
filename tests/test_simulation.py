"""Exercise the packaged Crazyflie assets and upstream PID without a GUI."""

import gymnasium
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def test_crazyflie_initializes_steps_and_closes_headlessly() -> None:
    target = np.array([0.0, 0.0, 0.5])
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=target.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    client = env.getPyBulletClient()
    try:
        assert isinstance(env, gymnasium.Env)
        assert p.getConnectionInfo(client)["connectionMethod"] == p.DIRECT
        assert p.getNumBodies(physicsClientId=client) == 2  # Plane and packaged Crazyflie.
        obs, info = env.reset(seed=0)
        assert obs.shape == (1, 20)
        assert env.action_space.shape == (1, 4)
        assert isinstance(info, dict)
        controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        for _ in range(env.CTRL_FREQ):
            rpm, _, _ = controller.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=target,
            )
            assert np.isfinite(rpm).all()
            obs, _, terminated, truncated, _ = env.step(rpm.reshape(1, 4))
            assert obs.shape == (1, 20)
            assert np.isfinite(obs).all()
            assert not terminated
            assert not truncated
        np.testing.assert_allclose(obs[0, :3], target, atol=0.05)
    finally:
        env.close()
    assert not p.isConnected(client)
