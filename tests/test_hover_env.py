"""The real headless simulator, Gymnasium contract, allocation and episode rules."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
import pytest
from stable_baselines3.common.env_checker import check_env

from aerorl.envs.hover_env import HoverEnv
from scripts.evaluate_hover import evaluate_episode, summarize_metrics


@pytest.fixture
def env():
    environment = HoverEnv()
    try:
        yield environment
    finally:
        environment.close()


def test_sb3_compatibility_and_seeded_reset(env):
    check_env(env)
    first, first_info = env.reset(seed=4)
    second, second_info = env.reset(seed=4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_info["state"], second_info["state"])
    third, _ = env.reset(seed=5)
    assert not np.array_equal(first, third)
    assert env.observation_space.contains(first)
    assert first.shape == (19,)
    assert env.action_space.shape == (4,)


def test_action_scaling_and_allocation_integration(env):
    env.reset(seed=0)
    np.testing.assert_allclose(env.action_to_wrench(np.zeros(4)), [env.weight, 0, 0, 0])
    high = env.action_to_wrench(np.ones(4))
    np.testing.assert_allclose(high, np.r_[1.6 * env.weight, np.diag(env.sim.J) * [8, 8, 4]])
    low = env.action_to_wrench(-np.ones(4))
    assert low[0] == pytest.approx(0.4 * env.weight)
    obs, reward, _, _, info = env.step(np.array([0.2, -0.1, 0.3, -0.2]))
    np.testing.assert_allclose(
        info["requested_wrench"], env.action_to_wrench([0.2, -0.1, 0.3, -0.2])
    )
    np.testing.assert_allclose(info["achieved_wrench"], info["requested_wrench"], atol=1e-14)
    assert env.sim.step_counter == 4  # 240 Hz physics / 60 Hz policy.
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    np.testing.assert_allclose(obs[-4:], [0.2, -0.1, 0.3, -0.2])


def test_explicit_saturation_diagnostics(env):
    env.reset(seed=1)
    _, reward, _, _, info = env.step_wrench([1.2 * env.allocator.max_thrust, 0.01, 0.01, 0.01])
    assert info["allocation_limited"]
    assert info["torque_scale"] == 0
    assert info["saturated_motors"].all()
    np.testing.assert_allclose(
        info["achieved_wrench"], [env.allocator.max_thrust, 0, 0, 0], atol=1e-14
    )
    assert info["wrench_error"][0] < 0
    assert np.isfinite(reward)


@pytest.mark.parametrize("bad", [[0, 0, 0], [np.nan, 0, 0, 0], [2, 0, 0, 0]])
def test_invalid_actions_are_rejected(env, bad):
    env.reset()
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.steps == 0


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"position": [0, 0, 0.1]}, "unsafe_altitude"),
        ({"position": [0, 0, 2.6]}, "unsafe_altitude"),
        ({"position": [1.6, 0, 1]}, "position_limit"),
        ({"rpy": [1.3, 0, 0]}, "attitude_limit"),
    ],
)
def test_unsafe_initial_states_terminate(env, options, reason):
    env.reset(seed=0, options=options)
    observation, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["termination_reason"] == reason
    assert not info["is_success"]
    assert np.isfinite(reward) and np.isfinite(observation).all()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_ground_contact_termination(env, monkeypatch):
    env.reset(seed=0)
    monkeypatch.setattr(p, "getContactPoints", lambda *args, **kwargs: (("contact",),))
    _, _, terminated, _, info = env.step(np.zeros(4))
    assert terminated
    assert info["termination_reason"] == "ground_contact"


def test_non_finite_simulator_state_terminates_with_finite_return(env, monkeypatch):
    env.reset(seed=0)
    original_step = env.sim.step

    def bad_step(action):
        obs, *rest = original_step(action)
        obs[0, 0] = np.nan
        return obs, *rest

    monkeypatch.setattr(env.sim, "step", bad_step)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["invalid_state"]
    assert info["termination_reason"] == "non_finite_state"
    assert np.isfinite(obs).all() and np.isfinite(reward)


def test_hover_equilibrium_and_time_limit():
    env = HoverEnv(randomization="none", episode_seconds=1)
    try:
        env.reset(seed=0)
        for i in range(60):
            obs, reward, terminated, truncated, info = env.step(np.zeros(4))
            assert not terminated
            assert truncated == (i == 59)
            assert reward == pytest.approx(1 / 60)
            np.testing.assert_allclose(info["state"][:3], [0, 0, 1], atol=1e-9)
        assert info["is_success"]
    finally:
        env.close()


def test_short_random_rollout_and_smoothness_cost(env):
    env.reset(seed=9)
    random = np.random.default_rng(3)
    for _ in range(120):
        action = random.uniform(-0.1, 0.1, 4)
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert info["reward_costs"]["smoothness"] >= 0
        if terminated or truncated:
            break


def test_paired_baselines_share_initial_conditions():
    initial = []
    metrics = []
    for controller, frequency in (("constant", 60), ("pid", 240)):
        env = HoverEnv(control_freq=frequency, episode_seconds=3, randomization="easy")
        try:
            _, metric, state = evaluate_episode(env, controller, None, seed=123)
            initial.append(state)
            metrics.append(metric)
        finally:
            env.close()
    for key in initial[0]:
        np.testing.assert_array_equal(initial[0][key], initial[1][key])
    assert metrics[1]["position_rmse_m"] < metrics[0]["position_rmse_m"]


def test_smoke_training_and_resume_write_models(tmp_path):
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv/bin/python"
    script = root / "scripts/train_hover.py"
    first, second = tmp_path / "first", tmp_path / "second"
    common = [str(python), str(script), "--smoke", "--total-timesteps", "128"]
    subprocess.run(common + ["--run-dir", str(first)], cwd=root, check=True, capture_output=True)
    subprocess.run(
        common + ["--run-dir", str(second), "--resume", str(first / "final_model.zip")],
        cwd=root,
        check=True,
        capture_output=True,
    )
    result = json.loads((second / "training_result.json").read_text())
    assert result["total_model_timesteps"] == 256
    assert result["additional_timesteps"] == 128
    assert (second / "final_model.zip").is_file()
    invalid_directory = tmp_path / "invalid"
    invalid = subprocess.run(
        [str(python), str(script), "--total-timesteps", "0", "--run-dir", str(invalid_directory)],
        cwd=root,
        capture_output=True,
    )
    assert invalid.returncode == 2
    assert not invalid_directory.exists()


def test_summary_reports_worst_deviation_separately_from_mean():
    record = dict.fromkeys(
        (
            "duration_s",
            "reward",
            "position_rmse_m",
            "final_second_rmse_m",
            "speed_rms_m_s",
            "angular_speed_rms_rad_s",
            "limited_fraction",
            "motor_bound_fraction",
        ),
        0.1,
    )
    summary = summarize_metrics(
        [
            {**record, "success": False, "max_position_error_m": 1.5},
            {**record, "success": True, "max_position_error_m": 0.1},
        ]
    )
    assert summary["max_position_error_m"] == 1.5
    assert summary["mean_episode_max_position_error_m"] == 0.8
    assert summary["success_rate"] == 0.5
