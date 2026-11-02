"""Validate the equations against the installed CF2X and run the real simulator."""

import csv
import json

import numpy as np
import pybullet as p
import pytest

from aerorl.control.actuation import cf2x_wrench, hover_rpm
from scripts.drone_fundamentals import (
    EXPERIMENTS,
    PATTERNS,
    make_environment,
    rotor_positions_body,
    run_experiment,
    save_results,
)


@pytest.fixture
def env():
    simulation = make_environment()
    client = simulation.getPyBulletClient()
    try:
        yield simulation
    finally:
        simulation.close()
    assert not p.isConnected(client)


def test_hover_equation_matches_upstream_and_loaded_assets(env):
    rpm = hover_rpm(env.M, env.G, env.KF)
    assert rpm == pytest.approx(env.HOVER_RPM)
    assert rpm == pytest.approx(14468.4291835)
    assert env.MAX_RPM / rpm == pytest.approx(1.5)
    positions = rotor_positions_body(env)
    np.testing.assert_allclose(
        positions,
        [[0.028, -0.028, 0], [-0.028, -0.028, 0], [-0.028, 0.028, 0], [0.028, 0.028, 0]],
        atol=1e-9,
    )
    wrench = cf2x_wrench(np.full(4, rpm), positions, env.KF, env.KM)
    np.testing.assert_allclose(wrench, [env.M * env.G, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize(
    "values", [(0, 9.8, 1e-10), (0.027, -1, 1e-10), (0.027, 9.8, 0), (np.nan, 9.8, 1e-10)]
)
def test_hover_rejects_invalid_parameters(values):
    with pytest.raises(ValueError, match="finite and positive"):
        hover_rpm(*values)


def test_upstream_clips_motor_commands_before_applying_forces(env):
    requested = np.array([-100.0, env.MAX_RPM * 2, env.HOVER_RPM, 0.0])
    obs, *_ = env.step(requested.reshape(1, 4))
    np.testing.assert_allclose(obs[0, 16:20], [0, env.MAX_RPM, env.HOVER_RPM, 0])
    positions = rotor_positions_body(env)
    requested_wrench = cf2x_wrench(requested, positions, env.KF, env.KM)
    applied_wrench = cf2x_wrench(obs[0, 16:20], positions, env.KF, env.KM)
    assert applied_wrench[0] < requested_wrench[0]


@pytest.mark.parametrize("axis,name", list(enumerate(("roll", "pitch", "yaw"))))
def test_wrench_sign_and_magnitude_match_initial_angular_acceleration(env, axis, name):
    rpm = env.HOVER_RPM * np.sqrt(1 + 0.02 * PATTERNS[name])
    wrench = cf2x_wrench(rpm, rotor_positions_body(env), env.KF, env.KM)
    assert wrench[0] == pytest.approx(env.M * env.G)
    assert wrench[axis + 1] > 0
    obs, *_ = env.step(rpm.reshape(1, 4))
    # Initially level and stationary: J * angular_acceleration = body torque.
    # This checks one 1/240 s physics step, before significant rotation or damping.
    expected_omega = np.linalg.solve(env.J, wrench[1:]) * env.CTRL_TIMESTEP
    np.testing.assert_allclose(obs[0, 13:16], expected_omega, rtol=0.005, atol=1e-8)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_each_experiment_runs_headlessly_and_records_expected_response(env, name):
    data, settings = run_experiment(env, name, duration=3)
    assert data["state"].shape == (720, 20)
    assert settings["gym_pybullet_drones_version"] == "2.2.0"
    assert settings["physics"] == "pyb"
    for key in (
        "state",
        "omega_body",
        "requested_rpm",
        "applied_rpm",
        "requested_wrench",
        "applied_wrench",
    ):
        assert np.isfinite(data[key]).all()
    np.testing.assert_allclose(data["time_s"] - data["command_time_s"], 1 / 240)
    assert np.all(data["applied_rpm"] >= 0)
    assert np.all(data["applied_rpm"] <= env.MAX_RPM)
    np.testing.assert_allclose(data["requested_rpm"], data["applied_rpm"])
    if name == "hover":
        np.testing.assert_allclose(data["state"][:, :3], np.tile([0, 0, 1], (720, 1)), atol=1e-8)
        assert not data["pid_active"].any()
    elif name in PATTERNS:
        axis = ("roll", "pitch", "yaw").index(name)
        sample = np.argmin(abs(data["time_s"] - 1.25))
        assert data["state"][sample, 7 + axis] > 0.01
    elif name.startswith("collective-"):
        sample = np.argmin(abs(data["time_s"] - 1.5))
        direction = 1 if name == "collective-up" else -1
        assert direction * data["state"][sample, 12] > 0.1
    else:
        assert data["pid_active"].all()
        assert data["state"][-1, 0] > 0.2
    if name not in ("hover", "position"):
        np.testing.assert_array_equal(
            data["pid_active"], data["command_time_s"] >= settings["recovery_start_s"]
        )


def test_position_maneuver_reaches_requested_position(env):
    data, _ = run_experiment(env, "position", duration=6)
    np.testing.assert_allclose(data["state"][-1, :3], [0.4, -0.2, 1.2], atol=0.05)
    assert np.linalg.norm(data["state"][-1, 10:13]) < 0.05
    # Position alone can hide a motor/rate limit cycle at insufficient control rates.
    tail = data["time_s"] > 5
    assert np.sqrt(np.mean(data["omega_body"][tail] ** 2)) < 0.02


@pytest.mark.parametrize("name,index", [("roll", 0), ("pitch", 1), ("yaw", 2)])
def test_attitude_pulse_is_visible_and_pid_recovers(env, name, index):
    data, settings = run_experiment(env, name, duration=6)
    raw = ~data["pid_active"]
    peak = np.rad2deg(abs(data["state"][raw, 7 + index])).max()
    assert 8 < peak < 30
    assert data["state"][:, 2].min() > 0.8
    np.testing.assert_allclose(data["state"][-1, :3], [0, 0, 1], atol=0.03)
    assert abs(np.rad2deg(data["state"][-1, 7 + index])) < 2
    assert settings["motion_summary"]["max_translation_m"] < 0.5


def test_pulse_magnitude_and_duration_are_configurable(env):
    first, _ = run_experiment(env, "roll", duration=3, pulse_magnitude=0.004, pulse_duration=0.15)
    second, settings = run_experiment(
        env, "roll", duration=3, pulse_magnitude=0.008, pulse_duration=0.3
    )
    assert np.count_nonzero(second["phase"] == "pulse") > np.count_nonzero(
        first["phase"] == "pulse"
    )
    assert second["applied_wrench"][:, 1].max() > first["applied_wrench"][:, 1].max()
    assert settings["pulse_duration_s"] == 0.3


def test_telemetry_csv_metadata_and_plot_are_saved(env, tmp_path):
    data, settings = run_experiment(env, "roll", duration=3)
    save_results(data, settings, tmp_path)
    with (tmp_path / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 720
    assert float(rows[-1]["time_s"]) == 3
    assert "p_body_rad_s" in rows[-1]
    assert "tz_applied_Nm" in rows[-1]
    assert rows[-1]["phase"] == "recovery-pid"
    saved_settings = json.loads((tmp_path / "parameters.json").read_text())
    assert saved_settings["hover_rpm"] == pytest.approx(env.HOVER_RPM)
    assert saved_settings["upstream_commit"]
    assert (tmp_path / "telemetry.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
