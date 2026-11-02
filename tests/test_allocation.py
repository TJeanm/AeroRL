"""Check allocation limits, forward consistency and actual stock CF2X responses."""

import csv
import json

import numpy as np
import pybullet as p
import pytest

from aerorl.control.actuation import cf2x_wrench
from aerorl.control.allocation import CF2XAllocator, allocation_matrix
from aerorl.utils.simulation import make_environment, rotor_positions_body
from aerorl.utils.telemetry import save_results
from scripts.motor_allocation import (
    EXPERIMENTS,
    make_allocator,
    plot_allocation_summary,
    run_experiment,
)


@pytest.fixture
def env():
    simulation = make_environment()
    try:
        yield simulation
    finally:
        client = simulation.getPyBulletClient()
        simulation.close()
        assert not p.isConnected(client)


def test_hover_and_random_feasible_roundtrips(env):
    allocator = make_allocator(env)
    positions = rotor_positions_body(env)
    result = allocator.allocate([env.M * env.G, 0, 0, 0], strict=True)
    np.testing.assert_allclose(result.rpm, env.HOVER_RPM)
    assert not result.limited
    random = np.random.default_rng(7)
    for rpm in random.uniform(0, env.MAX_RPM, (100, 4)):
        wrench = cf2x_wrench(rpm, positions, env.KF, env.KM)
        result = allocator.allocate(wrench, strict=True)
        np.testing.assert_allclose(result.rpm, rpm, atol=1e-7)
        np.testing.assert_allclose(result.achieved_wrench, wrench, atol=1e-14)
        np.testing.assert_allclose(result.wrench_error, 0, atol=1e-14)


@pytest.mark.parametrize(
    "thrust,torques",
    [
        (0, [0, 0, 0]),
        (1, [0, 0, 0]),
        (0.5, [1, -0.5, 0.75]),
        (-1, [1, 1, 1]),
        (2, [1, 1, 1]),
    ],
)
def test_collective_priority_direction_and_explicit_error(env, thrust, torques):
    allocator = make_allocator(env)
    requested = np.array([thrust * allocator.max_thrust, *np.array(torques) * 0.02])
    result = allocator.allocate(requested)
    assert np.all(result.rpm >= 0)
    assert np.all(result.rpm <= env.MAX_RPM)
    np.testing.assert_allclose(
        result.achieved_wrench[0], np.clip(requested[0], 0, allocator.max_thrust), atol=1e-14
    )
    np.testing.assert_allclose(
        result.achieved_wrench[1:], result.torque_scale * requested[1:], atol=1e-14
    )
    independently_achieved = cf2x_wrench(result.rpm, rotor_positions_body(env), env.KF, env.KM)
    np.testing.assert_allclose(result.wrench_error, independently_achieved - requested, atol=1e-14)
    if result.limited:
        assert result.saturated_motors.any()
        with pytest.raises(ValueError, match="infeasible wrench"):
            allocator.allocate(requested, strict=True)
    else:
        allocator.allocate(requested, strict=True)


def test_yaw_saturation_has_known_limit_and_maximal_scale(env):
    allocator = make_allocator(env)
    requested = np.array([env.M * env.G, 0, 0, 0.02])
    result = allocator.allocate(requested)
    expected_yaw = env.KM / env.KF * requested[0]
    assert result.limited and not result.collective_clipped
    assert result.torque_scale == pytest.approx(expected_yaw / 0.02)
    assert result.achieved_wrench[3] == pytest.approx(expected_yaw)
    np.testing.assert_allclose(result.rpm[[0, 2]], 0, atol=1e-3)
    assert result.saturated_motors.tolist() == [True, False, True, False]
    # A slightly larger scale violates a motor bound, even though T is unchanged.
    candidate = np.linalg.solve(
        allocator.matrix, np.r_[requested[0], requested[1:] * (result.torque_scale + 1e-6)]
    )
    assert candidate.min() < 0


def test_saturated_yaw_matches_actual_initial_response(env):
    result = make_allocator(env).allocate([env.M * env.G, 0, 0, 0.02])
    obs, *_ = env.step(result.rpm[None, :])
    actual = cf2x_wrench(obs[0, 16:20], rotor_positions_body(env), env.KF, env.KM)
    np.testing.assert_allclose(actual, result.achieved_wrench, atol=1e-14)
    assert obs[0, 15] == pytest.approx(result.achieved_wrench[3] / env.J[2, 2] / 240)


@pytest.mark.parametrize("axis,sign", [(i, s) for i in range(3) for s in (-1, 1)])
def test_both_torque_signs_match_simulated_initial_acceleration(env, axis, sign):
    allocator = make_allocator(env)
    command = np.array([env.M * env.G, 0.0, 0.0, 0.0])
    command[axis + 1] = sign * 6e-5
    result = allocator.allocate(command, strict=True)
    obs, *_ = env.step(result.rpm[None, :])
    expected = np.linalg.solve(env.J, command[1:]) * env.CTRL_TIMESTEP
    np.testing.assert_allclose(obs[0, 13:16], expected, rtol=0.005, atol=1e-8)
    assert sign * obs[0, 7 + axis] > 0


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 0.1])
def test_collective_changes_simulated_vertical_velocity(env, fraction):
    result = make_allocator(env).allocate([env.M * env.G * (1 + fraction), 0, 0, 0], strict=True)
    obs, *_ = env.step(result.rpm[None, :])
    assert obs[0, 12] == pytest.approx(fraction * env.G * env.CTRL_TIMESTEP, abs=1e-10)


def test_combined_wrench_matches_simulated_initial_acceleration(env):
    command = np.array([env.M * env.G * 1.05, 4e-5, -3e-5, 8e-5])
    result = make_allocator(env).allocate(command, strict=True)
    obs, *_ = env.step(result.rpm[None, :])
    np.testing.assert_allclose(
        obs[0, 13:16], np.linalg.solve(env.J, command[1:]) / 240, rtol=0.005, atol=1e-8
    )
    assert obs[0, 12] > 0


@pytest.mark.parametrize("bad", [[np.nan, 0, 0, 0], [1, np.inf, 0, 0], [1, 2, 3], [[1, 0, 0, 0]]])
def test_invalid_commands_are_rejected(env, bad):
    with pytest.raises(ValueError, match="finite"):
        make_allocator(env).allocate(bad)


def test_invalid_geometry_and_coefficients_are_rejected(env):
    positions = rotor_positions_body(env)
    for bad_positions, kf, km in [
        (np.zeros((4, 3)), env.KF, env.KM),
        (positions + [1, 0, 0], env.KF, env.KM),
        (positions, 0, env.KM),
        (positions, env.KF, -1),
    ]:
        with pytest.raises(ValueError):
            allocation_matrix(bad_positions, kf, km)
    with pytest.raises(ValueError):
        CF2XAllocator(positions, env.KF, env.KM, -1)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_all_demonstrations_run_and_compare_realized_wrench(env, name):
    data, settings = run_experiment(env, name, duration=3)
    assert data["state"].shape == (720, 20)
    assert np.isfinite(data["state"]).all()
    assert data["state"][:, 2].min() > 0.3
    np.testing.assert_allclose(data["requested_rpm"], data["applied_rpm"], atol=1e-8)
    np.testing.assert_allclose(
        data["wrench_error"], data["applied_wrench"] - data["requested_wrench"], atol=1e-14
    )
    limited = name in ("infeasible-torque", "infeasible-thrust", "negative-thrust")
    assert bool(data["limited"].any()) == limited
    assert settings["pulse_limited"] == limited
    if not limited:
        np.testing.assert_allclose(data["wrench_error"], 0, atol=1e-13)


def test_allocation_output_contains_limit_diagnostics(env, tmp_path):
    data, settings = run_experiment(env, "infeasible-torque", duration=3)
    save_results(data, settings, tmp_path)
    plot_allocation_summary([settings], tmp_path / "allocation_summary.png")
    with (tmp_path / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    pulse = next(row for row in rows if row["allocation_limited"] == "True")
    assert float(pulse["torque_scale"]) < 1
    assert abs(float(pulse["tz_error_Nm"])) > 0.001
    assert pulse["motor0_at_limit"] == "True"
    assert json.loads((tmp_path / "parameters.json").read_text())["pulse_limited"]
    assert (tmp_path / "allocation_summary.png").read_bytes().startswith(b"\x89PNG")
