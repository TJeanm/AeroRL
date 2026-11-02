# Motor allocation for body-wrench control

This milestone starts from AeroRL commit `78f81e2`. The installed source remains
gym-pybullet-drones **2.2.0**, upstream commit
`f83fbdd4ee1c07e1afa52694963f3a9a889a0f56`. Allocation is algebra in NumPy; PyBullet
and the existing Crazyflie model still integrate all motion. No PID gains, motor
coefficients, dynamics or 3D assets are changed. No RL environment is introduced.

## Run the demonstrations

From the repository root:

```sh
# Improved fundamentals: one stronger attitude pulse, then existing PID recovery.
.venv/bin/python scripts/drone_fundamentals.py --experiment all
.venv/bin/python scripts/drone_fundamentals.py --experiment roll --pulse-magnitude 0.008 --pulse-duration 0.3 --camera-distance 0.55

# All 13 physical-wrench allocation cases in one PyBullet GUI window.
.venv/bin/python scripts/motor_allocation.py --experiment all
.venv/bin/python scripts/motor_allocation.py --experiment yaw-negative
.venv/bin/python scripts/motor_allocation.py --experiment infeasible-torque

# Display-free checks and tests.
.venv/bin/python scripts/motor_allocation.py --experiment all --headless
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

The camera is now 0.55 m from the vehicle, rather than 1.5 m, and follows its
translation at 15 Hz without rotating with the drone. RGB body axes and motor
number overlays make attitude changes easier to see. `--camera-distance` changes
the distance in either script. Physics and control remain at 240 Hz.

In fundamentals, `--pulse-magnitude` is a fractional change of each motor's
**thrust/RPM squared**, not an RPM fraction. Default 0.008 is four times the old
0.002. The default pulse starts at 1 s and lasts 0.3 s. A 0.1 s coast shows that
rotation continues without torque; PID recovery starts at 1.4 s. There is no
attitude counterpulse. Collective experiments retain their paired ±10% thrust
pulses and recovery at 2 s. Position PID and open-loop hover are retained.
Changing `--duration` extends observation/recovery, not the pulse.

Every fundamentals experiment prints and saves maximum absolute roll, pitch and
yaw excursions in degrees, plus maximum Euclidean displacement from `[0,0,1]` m
in `motion_summary`. These maxima cover both excitation and PID recovery; a large
opposite recovery angle is included. Default raw excitation reaches about 18.5°
roll/pitch and 10.7° yaw, with full-maneuver peaks near 32.5° roll/pitch. The tests
check visibility, altitude and final recovery. Larger user-selected pulses can
exceed the upstream PID's recovery capability; this is not an aerobatics controller.

Allocation results are saved to
`results/motor_allocation/<timestamp>/<experiment>/`. Each case has the existing
eight-panel `telemetry.png`, SI-unit `telemetry.csv`, and `parameters.json`.
`allocation_summary.png` compares requested and achieved physical wrench across
all selected cases and shows torque scaling. No additional dependencies are used.

## Model geometry, numbering and frames

The existing [fundamentals notes](drone_fundamentals.md) describe the full model.
The source of truth for this allocator is the installed
[`cf2x.urdf`](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/assets/cf2x.urdf)
and
[`BaseAviary._physics()`](https://github.com/learnsyslab/gym-pybullet-drones/blob/f83fbdd4ee1c07e1afa52694963f3a9a889a0f56/gym_pybullet_drones/envs/BaseAviary.py).

Body +x is forward, +y left and +z up, with right-handed positive torques. World
gravity points down. Position/linear velocity and upstream angular velocity are
world-frame quantities; telemetry also converts angular velocity into body axes.

| Motor | Body x | Body y | Yaw reaction on body |
|---|---:|---:|---:|
| 0, front right | +0.028 m | −0.028 m | − |
| 1, rear right | −0.028 m | −0.028 m | + |
| 2, rear left | −0.028 m | +0.028 m | − |
| 3, front left | +0.028 m | +0.028 m | + |

`rotor_positions_body()` reads rotor COM locations from the loaded PyBullet model
and transforms them into body coordinates. The allocator uses these positions,
`env.KF = 3.16e-10 N/RPM²`, `env.KM = 7.94e-12 N m/RPM²`, and
`env.MAX_RPM ≈ 21702.644`. It does not substitute the slightly different nominal
`L/sqrt(2)`. Its matrix assumes centered, planar CF2X geometry with parallel +z
rotors and the motor ordering above. Invalid/singular geometry is rejected.

## Forward and inverse equations

Let `u_i = RPM_i²`. For rotor i at body position `(x_i,y_i,0)`:

```text
f_i = kF * u_i
roll moment  =  y_i * f_i
pitch moment = -x_i * f_i
yaw reaction = s_i * kM * u_i, with s = [-1,+1,-1,+1]

w = [T, tau_x, tau_y, tau_z]^T = A @ u

    [ kF       kF       kF       kF   ]
A = [ kF*y0    kF*y1    kF*y2    kF*y3]
    [-kF*x0   -kF*x1   -kF*x2   -kF*x3]
    [-kM       kM      -kM       kM   ]
```

For a feasible wrench the unique solution is `u = solve(A,w)` and
`RPM = sqrt(u)`. There is no signed RPM. Inverting RPM rather than RPM squared
would be incorrect because thrust is quadratic. For the symmetric X geometry,
with `d=0.028`, the first motor, for example, has:

```text
u_0 = T/(4*kF) - tau_x/(4*d*kF) - tau_y/(4*d*kF) - tau_z/(4*kM)
```

The implementation solves the matrix from actual geometry rather than hardcoding
this formula. `cf2x_wrench()` remains an independent forward cross-product
calculation used to check the inverse and reconstruct actual simulator commands.

Hover is `w=[m*g,0,0,0] = [0.2646,0,0,0]`: each motor is 14468.429 RPM.
Positive roll increases motors 2/3 relative to 0/1; positive pitch increases 1/2
relative to 0/3; positive yaw increases 1/3 relative to 0/2. Negative torques reverse
these patterns. Changing thrust accelerates the vehicle vertically near level;
torque first changes angular velocity, then attitude. Tilt then redirects thrust.

## Saturation policy and diagnostics

The feasible wrench set is a coupled set, not four independent physical limits:
every motor must satisfy `0 <= u_i <= u_max = MAX_RPM²`. Independently clipping an
unconstrained inverse generally changes both thrust and torque direction.

`CF2XAllocator.allocate()` instead implements an explicit collective-priority rule:

1. Bound requested thrust to `T_b = clip(T, 0, 4*kF*u_max)`. Negative thrust is
   physically unavailable; excess positive thrust is also limited and reported.
2. Build the zero-torque baseline `u_b = [T_b/(4*kF)] * 4`.
3. Solve `delta_u = solve(A, [0,tau_x,tau_y,tau_z])`.
4. Select the largest `alpha` in `[0,1]` for which
   `0 <= u_b + alpha*delta_u <= u_max` for **every** motor.
5. Return `sqrt(u_b + alpha*delta_u)` and calculate the achieved wrench/error.

For a positive increment, a motor bounds alpha by `(u_max-u_b)/delta_u`; for a
negative increment, by `-u_b/delta_u`. The minimum bound across four motors gives
the answer. Code uses fractions of `u_max` for these comparisons. A final numerical
clamp only removes floating-point boundary roundoff, after the analytic constraint
has already been enforced.

This retains bounded collective thrust and preserves requested torque direction
and ratios, reducing all torques by the same alpha. It is not a weighted least
squares optimizer and does not prioritize roll/pitch over yaw. At zero or maximum
collective, there is no differential-thrust headroom, so nonzero torques scale to
zero. A future controller may choose a different priority policy.

The returned `AllocationResult` contains:

- `rpm`, `squared_rpm`, `requested_wrench`, `achieved_wrench`;
- `wrench_error = achieved_wrench - requested_wrench`, in N and N m;
- `limited`, `collective_clipped`, and `torque_scale`;
- `saturated_motors`: motors at the lower or upper bound.

A feasible command exactly on a boundary can have saturated motors while
`limited=False`. When torque is zero, alpha can remain 1 even if thrust was
clipped, so callers must inspect `limited`/`collective_clipped` and the wrench
error, not alpha alone. `strict=True` raises `ValueError` for infeasible requests.
Malformed, non-finite commands and invalid model parameters are rejected.

At hover collective, requested yaw torque 0.020 N m achieves only 0.006648494 N m.
Alpha is 0.3324247; motors 0 and 2 reach zero. The reported yaw error is
−0.013351506 N m. Requested thrust 0.71442 N becomes 0.59535 N, with −0.11907 N
error; requested −0.02646 N becomes zero, with +0.02646 N error.

## Reuse and normalized future policy commands

```python
from aerorl.control.allocation import CF2XAllocator
from aerorl.utils.simulation import rotor_positions_body

allocator = CF2XAllocator(rotor_positions_body(env), env.KF, env.KM, env.MAX_RPM)
result = allocator.allocate([env.M * env.G, 6e-5, 0.0, 0.0])
obs, *_ = env.step(result.rpm.reshape(1, 4))
# Inspect result.limited and result.wrench_error in any downstream caller.
```

The allocator's interface is **SI units**, independent of RL. Normalized policy
commands require an explicit dimensional scaling *before* allocation. This
milestone does not silently choose whether normalized thrust zero means hover,
half maximum thrust, or stopped motors. Likewise, independent normalized torque
limits cannot guarantee joint feasibility at a given thrust. The future policy
adapter must document its scaling and retain allocation diagnostics. No PPO or
Gymnasium action wrapper is implemented here.

## Demonstration cases and actual-response validation

The allocation script covers hover, ±10% collective, positive/negative roll,
pitch and yaw, a combined thrust/three-torque command, infeasible yaw, excessive
thrust and negative thrust. Feasible pulses last 0.3 s from t=1 s; recovery then
uses the existing `DSLPIDControl`. Hover remains allocated collective throughout.
PID recovery is also passed through the allocator: the equivalent wrench of its
already-feasible RPM command reproduces those RPMs without modifying its gains.

Infeasible yaw is applied for one 1/240 s physics tick; excessive/negative thrust
for 0.12 s. This demonstrates real saturation without leaving a highly aggressive
command active long enough to flip or drop the vehicle. Those cases are actuator
limit probes, not attempts at sustained flight with infeasible commands.

At every step, telemetry compares the original requested wrench against the
independent `cf2x_wrench(obs[0,16:20], ...)` reconstruction. Added CSV columns
include four wrench errors, `allocation_limited`, `collective_clipped`,
`torque_scale`, and four motor-bound indicators. `requested_rpm` now means the
allocator's output command; there is no real unconstrained RPM when a solution
would require negative squared RPM. Command and observation timestamps remain
separate, as in the fundamentals notes.
Pulse boundaries are sampled on the 1/240 s control grid, so non-integral tick
durations end at the next control sample. CSV phase flags show the actual switch.

Tests independently check 100 random feasible round trips, hover, both torque
signs, combined commands, known saturation limits and maximal alpha, thrust
priority, torque ratios, strict rejection, invalid inputs, CSV/JSON/PNG output,
and all 13 headless demonstrations. Initial simulated angular velocity is compared
with `J^-1 @ achieved_torque * dt`, including saturated yaw; vertical velocity is
compared with `(T/m-g)*dt` near level hover.

An RPM-derived wrench is the applied rotor contribution, not a measured net body
wrench: it excludes gravity, contacts, default PyBullet damping and optional
aerodynamic effects. The `PYB` model still has no motor lag/battery dynamics. The
allocator is specific to the CF2X yaw pattern and model assumptions, and is not
calibrated for hardware. Large angular motions also expose the unchanged PID's
finite-difference Euler-rate limitations. Existing Gymnasium precision warnings
remain visible.

## Validation record

All **62 tests passed**, along with Ruff lint/format, `pip check` and Git
whitespace checks. The seven revised fundamentals GUI cases and all thirteen
allocation GUI cases completed on the Apple M3 OpenGL renderer with clean thread
shutdown. A separate headless allocation run also completed all thirteen cases.
Default allocation runs saved 960 samples per case. Feasible-case CSV residuals
were approximately at most 1.2e-16 N and 2e-18 N m. Minimum allocation-suite
altitude was 0.871 m during the negative-thrust probe. Fundamentals roll/pitch
reached about 18.5° before recovery and 32.5° over the full maneuver, with maximum
translation 0.306 m and minimum altitude 0.979 m. Yaw reached 11.0°. This validates
the simulator interface; it is not evidence of physical flight accuracy.
