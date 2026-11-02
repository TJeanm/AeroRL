"""Load and inspect the existing upstream CF2X; no custom simulation physics."""

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


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


def configure_camera(env: CtrlAviary, title: str, distance: float) -> None:
    """Close three-quarter view with motor numbers and body axes as debug overlays."""
    client = env.getPyBulletClient()
    follow_camera(env, [0, 0, 1], distance)
    drone = int(env.getDroneIds()[0])
    for i, offset in enumerate(rotor_positions_body(env)):
        p.addUserDebugText(
            str(i),
            (offset + [0, 0, 0.025]).tolist(),
            parentObjectUniqueId=drone,
            textSize=1.3,
            textColorRGB=[0, 0, 0],
            physicsClientId=client,
        )
    for axis, color in zip(np.eye(3), np.eye(3), strict=True):
        p.addUserDebugLine(
            [0, 0, 0],
            (0.10 * axis).tolist(),
            color.tolist(),
            parentObjectUniqueId=drone,
            lineWidth=3,
            physicsClientId=client,
        )
    p.addUserDebugText(
        title, [0, 0, 1.17], textSize=1.4, textColorRGB=[0, 0, 0], physicsClientId=client
    )


def follow_camera(env: CtrlAviary, position, distance: float) -> None:
    """Follow translation while keeping camera orientation fixed in the world."""
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=45,
        cameraPitch=-18,
        cameraTargetPosition=position,
        physicsClientId=env.getPyBulletClient(),
    )
