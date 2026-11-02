"""Check that the installed project and numerical/learning stack can load."""

import importlib

import numpy as np
import torch


def test_project_and_dependencies_import() -> None:
    for name in (
        "aerorl",
        "aerorl.envs",
        "aerorl.control",
        "aerorl.utils",
        "pybullet",
        "gym_pybullet_drones",
        "gymnasium",
        "stable_baselines3",
    ):
        assert importlib.import_module(name) is not None


def test_torch_numpy_interoperability() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal((torch.from_numpy(values) * 2).numpy(), values * 2)
