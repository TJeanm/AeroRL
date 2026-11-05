# PPO hover and stabilization

This is the first AeroRL learning task. The environment uses the installed
gym-pybullet-drones 2.2.0 CF2X, `CtrlAviary` and `Physics.PYB`. It adds a task,
normalized action interface and reward around the existing simulator, not drone
dynamics. The existing motor allocator is unchanged. There are no gates, tracks
or racing rewards. The README is unchanged.

## Markov decision process

The fixed world-frame target is `[0,0,1]` m, with zero velocity and identity
attitude as the desired equilibrium. The transition is four existing PyBullet
physics updates at 240 Hz, with one motor command held constant between policy
updates at **60 Hz**. The installed simulator checks divisibility of frequencies;
`240/60=4` is supported. PPO does not execute at 240 Hz.

The observation is a float32 vector of **19 elements**:

| Entries | Quantity | Frame and normalization |
|---|---|---|
| 0–2 | Position minus target | World, divided by 1 m |
| 3–5 | Linear velocity | World, divided by 2 m/s |
| 6–8 | Direction of body x | Three entries of the body-to-world rotation matrix |
| 9–11 | Direction of body y | Three entries of the body-to-world rotation matrix |
| 12–14 | Angular velocity | Body, divided by 5 rad/s |
| 15–18 | Previous normalized action | Each in `[-1,1]` |

Body z is the cross product of the two observed rotation columns. Thus these six
orientation entries encode the entire orientation without Euler/quaternion sign
discontinuities. PyBullet's quaternion is xyzw and its reported angular velocity
is world-frame; the environment transforms it with `R.T @ omega_world`.

Position, velocity, full orientation and angular velocity describe the rigid-body
state needed by this task. There is no hidden motor lag in this simulator path.
The previous action is included because the smoothness reward depends on it.
Observations are scaled rather than clipped, avoiding aliasing out-of-bounds
physical states into apparently safe observations. Position/velocity/rate space
bounds remain infinite; rotation entries and previous actions are bounded by one.

The **four actions** are `[a_T,a_roll,a_pitch,a_yaw]`, each in `[-1,1]`. They are
physical thrust/torque commands after scaling, not desired positions or rates:

```text
T = m*g*(1 + 0.6*a_T)
[tau_x,tau_y,tau_z] = diag(J) * [8,8,4] * [a_roll,a_pitch,a_yaw]
```

Zero action is hover equilibrium, not stopped motors. Thrust spans 0.10584–0.42336
N; torque magnitudes span 0.000112, 0.000112 and 0.0000868 N m. Torque limits derive
from model inertia and modest nominal angular acceleration limits (rad/s²), not
the much larger actuator extremes. This leaves thrust/torque headroom and reduces
initial exploration instability. It does not stabilize the drone by itself.

The action pipeline is:

```text
SB3 PPO -> four normalized actions -> SI wrench -> existing CF2XAllocator
        -> four RPM commands -> existing CtrlAviary/PyBullet -> observation
```

There is **no PID, attitude controller or rate controller** in `HoverEnv.step()`.
The allocator retains bounded collective and uniformly scales torque when needed;
requested/achieved wrench, residual, scale and motor-bound flags are returned in
`info`. Normal policy action limits are conservative enough that allocation
limiting is uncommon. A separate, explicitly named `step_wrench()` diagnostic path
allows physical-unit baseline commands and tests of infeasible requests. It does
not instantiate a controller. See [motor allocation](motor_allocation.md).

## Reward and episode rules

Let e be position minus target in m, v world velocity in m/s, R body-to-world
rotation, omega body rate in rad/s, a the normalized command and delta_a its change.

```text
c = 4*||e||² + 0.5*||v||² + 0.5*(3-trace(R))/2 + 0.02*||omega||²
    + 0.01*mean(a²) + 0.01*mean(delta_a²)

reward = (1-c)*dt - 2*I(unsafe termination)
dt = 1/60 s for PPO
```

Position gets the largest weight. Velocity and angular-rate penalties encourage
settling instead of passing through the target. Rotation error encourages level
attitude and restoration of yaw. Small effort and action-change penalties
discourage needless commands and chatter. The living reward requires maintaining
hover over time; it is not a one-time target-reaching bonus. Multiplication by dt
keeps accumulated reward in simulated-time units. The finite unsafe-termination
penalty is separate from time-limit truncation. Cost terms are exposed in `info`.

Unsafe termination occurs for ground contact, z below 0.15 or above 2.5 m,
Euclidean position error above 1.5 m, tilt above 60°, or a non-finite simulator
state. A non-finite state terminates explicitly while returning the last finite
observation, preventing NaNs entering SB3; it is not treated as successful flight.
Invalid/non-finite actions are rejected before stepping. Episodes otherwise
truncate at **8 simulated seconds**, 480 PPO actions.

Surviving an episode is not sufficient for reported success. The entire last
second must also have position RMSE <0.15 m, speed RMS <0.2 m/s, body angular-speed
RMS <0.5 rad/s, maximum tilt <10°, and maximum absolute yaw <20°. Success is
evaluated from physical state, not from episode reward.

## Initial conditions

Reset uses Gymnasium's seeded local NumPy generator. The presets independently
sample uniform offsets in each axis:

| Preset | Position offset | World velocity | Euler attitude | Body rate |
|---|---|---|---|---|
| `none` | 0 | 0 | 0 | 0 |
| `easy` | ±0.04 m | ±0.03 m/s | ±3° | ±0.05 rad/s |
| `small` (default) | ±0.10 m | ±0.10 m/s | ±8° | ±0.20 rad/s |

Pose/velocity randomization uses public PyBullet reset functions on the existing
body; it does not change model parameters. Explicit `reset(options=...)` can set
`position`, `velocity`, `rpy` (rad) or `omega_body`. The zero-randomization preset
is a useful equilibrium/unit test, not convincing evidence of learned control.

## Training

Use the installed environment from the repository root:

```sh
# Short integration check: 2,048 steps, deliberately not a trained controller.
.venv/bin/python scripts/train_hover.py --smoke --run-dir results/rl_hover/my_smoke

# Initial easier perturbations, followed by small randomized starts.
.venv/bin/python scripts/train_hover.py --total-timesteps 100000 --randomization easy --seed 0 --run-dir results/rl_hover/my_easy
.venv/bin/python scripts/train_hover.py --resume results/rl_hover/my_easy/best_model.zip --total-timesteps 300000 --randomization small --seed 0 --run-dir results/rl_hover/my_small

# Or start directly on the default perturbation distribution.
.venv/bin/python scripts/train_hover.py --total-timesteps 300000 --seed 1 --run-dir results/rl_hover/my_seed1
```

Run directories must be new, so earlier models/logs are not overwritten. CPU is
explicitly selected; no MPS benefit is assumed. PyTorch uses one thread to avoid
overhead on the small MLP. Four sequential `DummyVecEnv` environments batch policy
inference without multiprocessing complexity. PPO is SB3's implementation:
64/64-unit actor and critic, learning rate 3e-4, gamma 0.995, GAE lambda 0.95,
512 steps/environment/rollout, batch 256, ten epochs, initial log standard
deviation −1. Smoke mode uses one environment and smaller updates.

The run directory contains `config.json`, `progress.csv`, per-environment episode
CSV logs, `starting_model.zip`, periodic checkpoints, `best_model.zip`,
`final_model.zip`, internal evaluation history and `training_result.json`.
No TensorBoard dependency is needed. Internal evaluation runs every 25,000 steps
on five randomized episodes; best model means best mean validation reward, not
necessarily successful stabilization. SB3 rounds requested steps to full rollout
batches; the result JSON records actual counts. Generated outputs remain ignored
under `results/`.

Resume restores SB3 weights/optimizer state and keeps its timestep counter; the
requested total is **additional** steps. Simulator/rollout RNG state is not saved,
so resume is functional rather than bitwise identical to an uninterrupted run.
Seeds support repeatable runs with the same code/platform; one seed is not enough
to establish general reliability. Changing action scaling or observation layout
invalidates old models even if their array dimensions happen to match.

## Paired evaluation and GUI

```sh
.venv/bin/python scripts/evaluate_hover.py --model results/rl_hover/my_small/best_model.zip --episodes 20 --seed 20000 --randomization small --output-dir results/rl_hover/my_evaluation
.venv/bin/python scripts/evaluate_hover.py --model results/rl_hover/my_small/best_model.zip --controllers ppo --episodes 3 --seed 21000 --gui

# Baselines alone, or a harder/longer generalization check.
.venv/bin/python scripts/evaluate_hover.py --controllers constant pid --episodes 20
.venv/bin/python scripts/evaluate_hover.py --model results/rl_hover/my_small/best_model.zip --episodes 20 --seed 22000 --episode-seconds 12
```

PPO evaluation uses deterministic inference. Every controller uses the same
per-episode initial seed; the script asserts that actual sampled initial
conditions match. Evaluation seeds 20000+ differ from training and internal
validation seeds. Do not pick the reported final model by repeatedly optimizing
against the held-out comparison seeds.

The constant baseline applies `[m*g,0,0,0]` at 60 Hz with no feedback. The PID
baseline explicitly creates upstream `DSLPIDControl` and runs at its validated
240 Hz, because its stock gains oscillated at lower update frequencies in prior
milestones. Its physical wrench passes through the same allocator. This comparison
uses identical simulated durations, physics and initial states, but deliberately
different controller rates; it is not a compute-matched contest. The PPO pipeline
never calls the PID baseline code. PID command effort may exceed the narrower RL
action scales, so reward comparisons are secondary to physical motion metrics.

Outputs include summary JSON, per-episode metrics CSV, exact initial conditions,
per-step telemetry CSV for every episode, plots for the first two episodes per
controller, and a comparison plot. Metrics include success/survival, reward,
position RMSE and maximum deviation, last-second RMSE, linear/angular speed RMS,
RPM, requested/achieved wrench and allocation/motor saturation. Euler angles are
logged for interpretation only; PPO never observes them.
Summary `max_position_error_m` is the worst deviation across all episodes;
`mean_episode_max_position_error_m` separately averages the individual maxima.

Episode RMSE averages only the time actually flown. A controller that crashes
early can have a misleadingly small RMSE, so always read it alongside duration,
termination reason and success rate. Last-second RMSE in a short failed episode
means its last available second, not successful end-of-horizon stabilization.

## Checks

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip check
```

Tests cover spaces, seeded randomization, normalized scaling, allocation and
saturation, four physics substeps, finite returns, reset/step contract, failure
termination, successful time-limit truncation, randomized headless rollouts,
SB3 `check_env`, paired baseline conditions, smoke training and resume. GUI access
is not needed for tests. Existing upstream Gymnasium precision warnings remain
visible. The PyBullet GUI is verified separately on the desktop.

## Results and limits

The initial smoke run completed 2,048 steps. The easy-start run completed 100,352
steps in 36.3 s; that model still achieved zero held-out successes. The next run
resumed its validation-selected checkpoint at 100,000 steps and trained another
301,056 steps, ending at 401,056 cumulative model steps in 93.0 s. The best model
selected **by internal validation reward** was the checkpoint at **275,000**
cumulative steps. The following held-out comparison evaluates that checkpoint,
not a model selected using the comparison seeds.

On seeds 20000–20019 with `small` randomization and eight-second episodes:

| Controller | Success | Mean survival | Mean episode position RMSE | Mean last-second RMSE | Mean reward |
|---|---:|---:|---:|---:|---:|
| PPO, 60 Hz | 20/20 | 8.00 s | 0.0495 m | 0.0191 m | 7.870 |
| Constant hover thrust, 60 Hz | 0/20 | 1.59 s | 0.6558 m | 0.8095 m | −4.313 |
| Upstream PID, 240 Hz | 20/20 | 8.00 s | 0.0209 m | 0.00246 m | 7.958 |

PPO's worst deviation across those episodes was 0.236 m. Its mean speed RMS was
0.0737 m/s and angular-speed RMS 0.1784 rad/s. No controller
hit allocation/motor limits in these comparisons. On a separate set of twenty
seeds, 22000–22019, PPO also achieved 20/20 successes over twelve-second episodes,
with mean episode position RMSE 0.0392 m and last-second RMSE 0.0162 m. These results
demonstrate stabilization and recovery from the specified initial perturbations;
the unchanged PID remains more accurate. PPO has a visible, decaying settling
transient and small residual position/yaw error. This is not evidence that PPO
outperforms PID or will recover from larger disturbances.

The saved local evidence is under `results/rl_hover/`:

- `smoke_seed0/`, `easy_seed0/`, `small_seed0/`: training logs, models and checkpoints;
- `evaluation_easy_model/`: the early model's 0/20 comparison;
- `evaluation_small_model/`: paired final comparison, telemetry and plots;
- `evaluation_longer/`: independent twelve-second evaluation;
- `gui_ppo/`: real-time PPO GUI demonstration on separate seeds.

Launch the actual trained checkpoint from this milestone:

```sh
.venv/bin/python scripts/evaluate_hover.py --model results/rl_hover/small_seed0/best_model.zip --controllers ppo --episodes 3 --seed 21000 --gui
.venv/bin/python scripts/evaluate_hover.py --model results/rl_hover/small_seed0/best_model.zip --episodes 20 --seed 23000
```

The full suite passed **79 tests**, with lint/format/dependency checks passing.
Only the existing upstream space-bound precision warnings remained. Tests also
exercise a fresh 128-step PPO save and another 128 steps after resume. The finite
observation fallback on simulator failure and inclusion of previous action avoid
two common RL implementation issues: NaNs entering optimization and unobserved
history dependence in a smoothness reward.
The three GUI PPO episodes on separate seeds 21000–21002 all succeeded and the
Apple M3 OpenGL renderer shut down cleanly.

This is a single stock simulator/model, without calibrated motor lag, batteries,
sensor noise, delay or optional aerodynamic effects. It does not establish
real-world or racing capability. Initial perturbations are small and there are
no mid-flight disturbances yet. There is only one training seed and a small
evaluation sample. Larger tilt/rate errors can leave the chosen torque range
unable to recover. Hardware transfer is outside this milestone. The recommended
next step is to repeat training across several seeds and evaluate controlled
mid-flight disturbances and larger initial offsets before adding a racing task.
