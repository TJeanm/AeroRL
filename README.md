# Autonomous quadrotor racing with deep reinforcement learning.

Work in progress.

Overview

AeroRL is a personal robotics project exploring reinforcement learning for autonomous quadrotor racing in simulation.

The objective is to train a quadrotor to fly through a sequence of gates while respecting the dynamics and control constraints of the vehicle.

The project has two main goals:

* understand the fundamentals of quadrotor dynamics and control;
* investigate how reinforcement learning can be used for low-level autonomous flight.

Existing simulation, vehicle models and standard control components are reused whenever possible. The focus is on the parts relevant to learning-based robotics: control interfaces, state representation, reward design, curriculum learning and evaluation.

Approach

The project uses an existing Crazyflie quadrotor model in PyBullet.

Before training the reinforcement learning policy, several basic experiments will be used to study:

* rotor thrust and motor mixing;
* collective thrust;
* roll, pitch and yaw control;
* hover equilibrium;
* body and world coordinate frames;
* actuator limits;
* classical cascaded PID control.

The reinforcement learning policy will then control the vehicle using collective thrust and body torques:

[T, τx, τy, τz]

The policy observes the quadrotor state together with the relative pose of the upcoming racing gates.

Training will progressively move from basic flight tasks to complete gate sequences.

Reinforcement Learning

The initial algorithm is Proximal Policy Optimization (PPO).

The learning problem will include:

* continuous control;
* progress toward the next gate;
* gate traversal;
* collision and flight constraints;
* time-efficient racing;
* curriculum learning.

Training and evaluation tracks will be kept separate to measure whether the learned policy can generalize beyond a single predefined course.

Stack

* Python
* PyTorch
* Gymnasium
* Stable-Baselines3
* PyBullet
* gym-pybullet-drones
* Crazyflie 2.x model

Evaluation

The final policy will be evaluated using metrics such as:

* track completion rate;
* gate traversal success rate;
* lap time;
* collision rate;
* performance on unseen tracks.

Additional robustness experiments may include changes in initial conditions and simulated disturbances.

Research Inspiration

This project is inspired by research on autonomous drone racing and learning-based agile flight, including:

* Song et al., Autonomous Drone Racing with Deep Reinforcement Learning, 2021
* Kaufmann et al., Champion-level Drone Racing using Deep Reinforcement Learning, Nature, 2023
* Song et al., Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning, Science Robotics, 2023
* Sun et al., Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles, ICRA, 2026

AeroRL is not intended to reproduce one specific paper. Instead, it uses these works as guidance for a smaller experimental project focused on quadrotor control and reinforcement learning.

Status

Early development.

Initial work focuses on understanding the quadrotor control interface and validating the simulation environment before reinforcement learning training.
