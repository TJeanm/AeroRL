# Drone Fundamentals: the installed Crazyflie simulator

This milestone explores actuation and feedback using one existing Crazyflie CF2X,
`CtrlAviary`, `Physics.PYB`, and the unchanged upstream `DSLPIDControl`. There is no
learning algorithm, racing environment, custom dynamics, controller or 3D asset.

## Source of truth

The installed distribution is **gym-pybullet-drones 2.2.0**, Git commit
`f83fbdd4ee1c07e1afa52694963f3a9a889a0f56`. Its installed `direct_url.json` confirms
the revision. These notes follow its executable code, rather than assuming that
every upstream docstring is current:

- [CF2X URDF: mass, inertia, rotor offsets and coefficients](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/assets/cf2x.urdf).
- [BaseAviary: constants, physics, timing and state vector](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/envs/BaseAviary.py).
- [CtrlAviary: actual motor action and clipping](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/envs/CtrlAviary.py).
- [BaseControl: mapping the state into controller arguments](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/control/BaseControl.py).
- [DSLPIDControl: cascaded controller and PWM mixing](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/control/DSLPIDControl.py).
- [CTBRControl](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/control/CTBRControl.py)
  and [BetaAviary](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/envs/BetaAviary.py):
  the separate Betaflight interface, not used in these experiments.

The local files are under
`.venv/lib/python3.12/site-packages/gym_pybullet_drones/`. Nothing in that directory
is modified. Each experiment's `parameters.json` records the installed version,
source commit, physical parameters, rotor positions and sampling frequencies.

## Launch commands

From the AeroRL repository root, run the full visual demonstration:

```sh
.venv/bin/python scripts/drone_fundamentals.py --experiment all
```

Each experiment lasts six simulated seconds, paced in real time in the GUI. The
same window is reset between experiments. Motor numbers and RGB body axes
(red x, green y, blue z) are debug overlays. To inspect one behavior at a time:

```sh
.venv/bin/python scripts/drone_fundamentals.py --experiment hover
.venv/bin/python scripts/drone_fundamentals.py --experiment collective-up
.venv/bin/python scripts/drone_fundamentals.py --experiment collective-down
.venv/bin/python scripts/drone_fundamentals.py --experiment roll
.venv/bin/python scripts/drone_fundamentals.py --experiment pitch
.venv/bin/python scripts/drone_fundamentals.py --experiment yaw
.venv/bin/python scripts/drone_fundamentals.py --experiment position
```

Display-free validation and a longer PID maneuver:

```sh
.venv/bin/python scripts/drone_fundamentals.py --experiment all --headless
.venv/bin/python scripts/drone_fundamentals.py --experiment position --duration 10
.venv/bin/python -m pytest -q
```

`--duration` must be at least three seconds. Pulse timings are fixed in seconds;
increasing duration extends the observation/recovery period, not the excitation.
`--output-dir /path/to/output` changes the result root. By default each invocation
creates `results/drone_fundamentals/<timestamp>/<experiment>/` containing:

- `telemetry.csv`: SI-unit data with explicit column names and control phases;
- `telemetry.png`: eight panels showing motion, attitude, motors and body wrench;
- `parameters.json`: model and experiment settings needed to interpret the run.

Plots are saved automatically; no Matplotlib window or additional dependency is
needed. The PyBullet window closes on completion or Ctrl-C. On interruption, only
already completed experiments are saved. Generated results are ignored by Git.

## Physical parameters

These are model parameters, not new estimates of a physical Crazyflie:

| Quantity | Installed value | Meaning |
|---|---:|---|
| Mass `M` | 0.027 kg | 27 g, base link; rotor links have zero mass |
| Gravity `G` | 9.8 m/s² | Simulator constant, not 9.81 |
| `GRAVITY` | 0.2646 N | Despite the name, this is **weight** `M * G` |
| `Jxx`, `Jyy` | 1.40 × 10⁻⁵ kg m² | Roll and pitch inertia |
| `Jzz` | 2.17 × 10⁻⁵ kg m² | Yaw inertia; inertia cross terms are zero |
| Nominal arm `L` | 0.0397 m | Nominal center-to-rotor distance |
| Actual rotor x/y offsets | ±0.028 m | Lever arms loaded by PyBullet from the URDF |
| `KF` | 3.16 × 10⁻¹⁰ N/RPM² | Per-motor thrust coefficient |
| `KM` | 7.94 × 10⁻¹² N m/RPM² | Per-motor yaw reaction coefficient |
| Thrust-to-weight limit | 2.25 | Used to calculate the motor RPM ceiling |
| `HOVER_RPM` | 14468.429 RPM | Level, stationary equilibrium per motor |
| `MAX_RPM` | 21702.644 RPM | `CtrlAviary` clips each motor to this maximum |
| `MAX_THRUST` | 0.59535 N | Four motors at maximum RPM |

`L / sqrt(2) = 0.028072139 m`, whereas the URDF uses 0.028 m, about 0.258% less.
`Physics.PYB` applies forces at the actual rotor COM locations. The separate
upstream `Physics.DYN` implementation and `MAX_XY_TORQUE` estimate use the nominal
arm. These experiments always use `PYB` and read loaded rotor positions for
telemetry. At extreme differential thrust, actual single-axis roll/pitch torque
can reach 0.0083349 N m; upstream's nominal value is 0.00835637 N m. Maximum yaw
torque is 0.00747956 N m. These maxima are not independently available at arbitrary
collective thrust: all outputs share the same four bounded motors.

## Coordinates and motor numbering

The world frame has z upward and gravity along negative z. World x/y are arbitrary
horizontal simulation axes, not a geographic heading. The body frame is right
handed; for interpretation, call +x forward, +y left and +z upward. Rotor forces
point along **body +z**, which tilts with the drone.

Let `R` map body vectors to world vectors. PyBullet supplies quaternion order
`[x, y, z, w]`; `getMatrixFromQuaternion()` gives `R`. PyBullet roll/pitch/yaw use
`R = Rz(yaw) Ry(pitch) Rx(roll)`. Positive angles/torques follow the right-hand rule:
positive roll raises the +y side; positive pitch lowers the +x side; positive yaw
turns +x toward +y when viewed from above.

| Motor index in action | Body x (m) | Body y (m) | Location | Body yaw torque sign |
|---|---:|---:|---|---|
| 0 | +0.028 | −0.028 | Front right | − |
| 1 | −0.028 | −0.028 | Rear right | + |
| 2 | −0.028 | +0.028 | Rear left | − |
| 3 | +0.028 | +0.028 | Front left | + |

These are simulator indices, not a claim about hardware connector numbering.
The yaw column is the reaction torque on the body; RPM is a nonnegative magnitude.
There is no signed motor-speed command or simulated rotor spin-up in this path.

For one drone, the observation has shape `(1, 20)`:

| Slice of `obs[0]` | Contents | Frame / units |
|---|---|---|
| `0:3` | Position | World, m |
| `3:7` | Quaternion | Body-to-world, xyzw |
| `7:10` | Roll, pitch, yaw | rad |
| `10:13` | Linear velocity | World, m/s |
| `13:16` | Angular velocity | **World**, rad/s, from `getBaseVelocity()` |
| `16:20` | Last clipped motor command | RPM, motors 0–3 |

Body angular velocity is `omega_body = R.T @ omega_world`. Its components `(p,q,r)`
are not generally equal to Euler-angle derivatives. CSV files contain both angular
velocity frames, raw Euler angles and the quaternion. Plots show angles in degrees
and unwrap yaw for readability; CSV angles remain the raw radians.

## From RPM to thrust and body torque

For motor i, with RPM magnitude `n_i`, the installed `_physics()` computes:

```text
f_i = KF * n_i²                    [N]
q_i = KM * n_i²                    [N m]
T   = f_0 + f_1 + f_2 + f_3
tau_x = d * (-f_0 - f_1 + f_2 + f_3)
tau_y = d * (-f_0 + f_1 + f_2 - f_3)
tau_z = KM * (-n_0² + n_1² - n_2² + n_3²)
d = 0.028 m for this loaded PYB model
```

Roll and pitch come from the moment arm `r_i × [0,0,f_i]`. PyBullet receives four
external rotor forces in `LINK_FRAME` and the summed yaw torque at the center link.
It then integrates the rigid body. `aerorl/control/actuation.py` only reconstructs
these algebraic quantities for interpreting/logging commands. It never integrates
motion or applies forces itself.

The equivalent motor-allocation matrix maps **squared RPM**, not RPM, into wrench:

```text
[T ]     [ KF       KF       KF       KF   ] [n_0²]
[tx]  =  [-d*KF    -d*KF     d*KF     d*KF ] [n_1²]
[ty]     [-d*KF     d*KF     d*KF    -d*KF ] [n_2²]
[tz]     [-KM       KM      -KM       KM   ] [n_3²]
```

This notation describes the mapping; the script does not add a general inverse
allocator or a new thrust/torque controller. Its small predefined pulses use:

```text
collective increase:   n_i = n_hover * sqrt(1 + 0.10) for every motor
positive roll:         n_i = n_hover * sqrt(1 + 0.002 * [-1,-1,+1,+1]_i)
positive pitch:        n_i = n_hover * sqrt(1 + 0.002 * [-1,+1,+1,-1]_i)
positive yaw:          n_i = n_hover * sqrt(1 + 0.002 * [-1,+1,-1,+1]_i)
```

The balanced attitude patterns keep total thrust exactly `mg` before saturation
while producing one body torque. The counterpulse reverses the signs inside the
square root. A 10% RPM increase would produce **21% more thrust**; the square root
is why our collective pulse produces precisely 10% more thrust instead.

The basic physical interpretation, away from contacts, is:

```text
m * acceleration_world = R @ [0, 0, T] - [0, 0, m*g] + other forces
J * omega_dot_body + omega_body × (J @ omega_body) = tau_body + other torques
```

These are explanatory rigid-body balance equations, not a replacement solver.
PyBullet still handles inertia, rotation, damping and contacts. Near level hover,
extra collective gives approximately `a_z = (T-mg)/m`; a +10% thrust pulse initially
gives +0.98 m/s². Initially stationary, angular acceleration is approximately
`J^-1 @ tau`. A positive roll tilts thrust toward world −y; positive pitch tilts it
toward world +x. Tilt reduces vertical thrust to approximately `T cos(tilt)`.

### Hover equilibrium and saturation

At rest, level, with four equal motors:

```text
4 * KF * n_hover² = m*g
n_hover = sqrt(m*g / (4*KF)) = 14468.429 RPM
n_max = sqrt(2.25*m*g / (4*KF)) = 1.5*n_hover
```

This balances forces; it does **not** stabilize position or attitude. Returning
to equal hover RPM does not remove a velocity, angular displacement or existing
tilt. The reverse pulse approximately brakes the corresponding rate. It does not
automatically return the drone to its starting pose.

The actual public `CtrlAviary` motor interface is:

```python
obs, reward, terminated, truncated, info = env.step(np.array([[rpm0, rpm1, rpm2, rpm3]]))
```

Each motor is clipped independently to `[0, MAX_RPM]` by `_preprocessAction()`.
Despite integer/dictionary wording in parts of its docstrings, the implementation
accepts floating-point RPM arrays. The action is held for the physics substeps;
`obs[0,16:20]` reports that clipped command, not a separate measured motor state.
Saturation changes both collective and differential torque. Requested and applied
wrenches are therefore logged separately. Nominal demo pulses do not saturate;
automated tests explicitly exercise negative and excessive RPM commands.

There is **no public `[T_N, tau_x_Nm, tau_y_Nm, tau_z_Nm]` action in `CtrlAviary`**.
The installed RL action enum also has no torque action; references to thrust/torque
in its generic docstrings are insufficient evidence of a supported interface.
`CTBRControl` returns collective specific thrust (m/s²; 9.8 at hover) and body rates
(rad/s) for the `BetaAviary`/Betaflight SITL path. Rates are not torques. That path
requires additional SITL executables and is not initialized here.

## Existing PID architecture

The position maneuver calls only the upstream public method:

```python
rpm, position_error, yaw_error = controller.computeControlFromState(
    control_timestep=env.CTRL_TIMESTEP,
    state=obs[0],
    target_pos=np.array([0.4, -0.2, 1.2]),
)
```

1. **Position loop.** World-frame position and velocity errors form a desired
   force vector: `F_des = Kp*position_error + Ki*integral_error + Kd*velocity_error
   + [0,0,mg]`. Integral state is clipped to ±2 (z further to ±0.15). Position-loop
   gains for x/y/z are `Kp=[.4,.4,1.25]`, `Ki=[.05,.05,.05]`, `Kd=[.2,.2,.5]`.
2. **Thrust and attitude targets.** Collective is the nonnegative projection of
   `F_des` onto the current body z axis. Desired body z follows `F_des`; desired
   yaw specifies the heading. Roll and pitch are computed from the desired force,
   so `target_rpy` is not a direct roll/pitch command through this position API.
3. **Attitude loop.** A rotation-matrix error, integral error and finite-difference
   Euler-rate error form three correction signals. Gains are
   `Kp=[70000,70000,60000]`, `Ki=[0,0,500]`, `Kd=[20000,20000,12000]`.
   The rotation error is extracted from `R_des.T @ R - R.T @ R_des` without a 1/2
   factor. The integral accumulates its negative and is clipped to ±1500, with
   roll/pitch further limited to ±1. The input `cur_ang_vel` is currently unused.
   `target_rpy_rates` means desired Euler-angle rates, not body rates or ACRO mode.
4. **PWM mixer and output.** Correction signals are clipped to ±3200 and mixed
   with a collective PWM value using the CF2X matrix below. These internal signals
   are **not physical torques in N m**, despite the variable name `target_torques`.

```text
collective_pwm = (sqrt(T_projected/(4*KF)) - 4070.3) / 0.2685

                 [-0.5, -0.5, -1]
MIXER_MATRIX =   [-0.5, +0.5, +1]
                 [+0.5, +0.5, -1]
                 [+0.5, -0.5, +1]

pwm = clip(collective_pwm + MIXER_MATRIX @ corrections, 20000, 65535)
rpm = 0.2685 * pwm + 4070.3
```

PID output is limited to **9440.3–21666.4475 RPM**, a narrower range than raw
`CtrlAviary` commands. In particular, PID cannot command stopped motors. It does
not expose physical thrust/torque targets in its public return value. For PID
phases our wrench traces are explicitly **equivalent values derived from returned
RPM**, not private PID correction signals or force-sensor measurements.

The controller constructs target rotations using SciPy intrinsic `XYZ` Euler
angles, while observed PyBullet Euler angles use the convention described above.
It differentiates observed angles directly, without unwrapping at ±pi. Avoid
interpreting its derivative term as a measured body-rate loop; these small
experiments avoid large rotations and yaw wrap crossings.

## Timing and experiment phases

`CtrlAviary` defaults to 240 Hz for both physics and control. This script explicitly
uses **240 Hz physics and 240 Hz commands**: one physics step per command, a
physics timestep of 1/240 s and a control timestep of 1/240 s. `pyb_freq` must be
divisible by `ctrl_freq`. The PID receives the **control** timestep. GUI pacing
uses upstream `sync()`; headless runs advance the same simulated time faster.

The earlier infrastructure smoke test uses 48 Hz control. Comparing the unchanged
PID at 48, 120 and 240 Hz revealed persistent motor and angular-rate oscillations
at 48 Hz during the position maneuver. This demonstration uses the upstream
default 240 Hz control rate and smaller attitude pulses for moderate recovery
motion. Controller gains are unchanged. Position convergence alone did not catch
the oscillation; a test also checks final body-rate RMS.

| Experiment | From 0 to 1 s | Excitation | From 2 s onward |
|---|---|---|---|
| `hover` | Equal equilibrium RPM | Same equilibrium command throughout | Open loop throughout |
| `collective-up` | Equilibrium RPM | +10% thrust at 1–1.5 s; −10% at 1.5–2 s | Existing PID returns to initial position |
| `collective-down` | Equilibrium RPM | −10% thrust at 1–1.5 s; +10% at 1.5–2 s | Existing PID returns to initial position |
| `roll` | Equilibrium RPM | Positive roll pulse at 1–1.25 s, reverse at 1.25–1.5 s, coast to 2 s | Existing PID recovers |
| `pitch` | Equilibrium RPM | Same timing, positive pitch pattern | Existing PID recovers |
| `yaw` | Equilibrium RPM | Same timing, positive yaw pattern | Existing PID recovers |
| `position` | PID holds `[0,0,1]` | At 1 s step target to `[.4,-.2,1.2]` m | PID holds new target |

Every experiment starts level at `[0,0,1]` m with zero velocity. Each gets a fresh
PID instance and a simulator reset. Recovery targets `[0,0,1]` m and zero yaw. It
uses the existing controller unchanged, not a new recovery controller. The CSV
`phase` and `pid_active` columns identify the switch. Plot references appear only
while PID is active; the raw hover/pulse phases have no position feedback.

Commands take effect at `command_time_s`; state is sampled after the physics
step at `time_s = command_time_s + 1/240`. Requested/applied RPM and wrench traces
are plotted against command time, motion against observation time. Wrench is in
the body frame, excludes gravity, and is not the net world force. Body angular
velocity is transformed using the orientation at the observation timestamp.

## What to look for and model limitations

- **Hover:** thrust equals weight and the initial pose remains stationary. Perfect
  initial conditions make open-loop hover look stable; it has no restoring feedback.
- **Collective:** thrust changes velocity first; position accumulates the motion.
  The reversed pulse brakes it; PID later returns the position to its target.
- **Roll/pitch:** torque changes angular rate first, then attitude. Tilt redirects
  thrust and produces horizontal motion and some altitude loss even when `T=mg`.
- **Yaw:** alternating motor reaction torques turn the body with little translation
  under these symmetric, level conditions.
- **Position PID:** the controller creates roll/pitch to accelerate sideways, then
  counteracts the motion and returns toward level at its new target.

`Physics.PYB` does not enable upstream optional rotor drag, ground effect or
downwash. Default PyBullet damping is still present: the upstream block that
would remove it is commented out. It is not a calibrated aerodynamic drag model.
Motors have no battery model, motor lag or separate rotor-speed dynamics. Contacts
use the existing URDF collision cylinder, not detailed propeller geometry. These
experiments start above the ground and use small perturbations.

`CtrlAviary` supplies a dummy reward and always-false termination/truncation flags;
it will not stop an unstable flight or ground contact on its own. Our experiments
are time limited and reject non-finite state but do not add a flight envelope or
collision safety system. Larger perturbations are outside this demonstration.
The existing Gymnasium float64-to-float32 space-bound warnings remain visible.

Tests check hover and limits against the installed model, compare predicted
initial torque-induced angular acceleration with PyBullet, execute every
experiment without a GUI, check position convergence and verify CSV/JSON/PNG
output. They are simulation checks, not validation against real flight data.

## Validation of this milestone

On the local Apple M3, all **21 tests passed**, along with Ruff lint/format,
`pip check` and Git whitespace checks. All seven six-second GUI experiments
completed with the Apple M3 OpenGL renderer and clean thread shutdown. Each saved
1,440 samples. The position maneuver ended at approximately
`[0.417167, -0.208376, 1.202739]` m, 0.0193 m from its target; body-rate RMS over
the last second was 0.000328 rad/s. Hover remained at `[0,0,1]` m. The remaining
test warnings are upstream Gymnasium space-bound precision warnings described
above. No physical hardware validation is implied.
