#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""JVRC-1 humanoid standing on two feet and reaching with a hand."""

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask, LowAccelerationTask
from pink.visualization import start_meshcat_visualizer
from pink.utils import get_root_joint_dim
import time

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    )


class WavingPose:
    """Moving target to the wave the right hand."""

    def __init__(self, init: pin.SE3):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init = init

    def at(self, t):
        """Get waving pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        R = T.rotation
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        T.rotation = R
        T.translation[0] += 0.2
        T.translation[1] += -0.1 + 0.2 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        T.translation[2] += 0.2 * np.sin(8.0 * t)
        return T


if __name__ == "__main__":
    robot = load_robot_description(
        "jvrc_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    wrist_frame = viewer["right_wrist_pose"]
    meshcat_shapes.frame(viewer["pelvis_pose"])
    meshcat_shapes.frame(wrist_frame)

    meshcat_shapes.frame(viewer["left_foot_target_pose"])
    meshcat_shapes.frame(viewer["right_foot_target_pose"])

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_foot_task = FrameTask(
        "l_ankle", position_cost=5.0, orientation_cost=5.0
    )
    right_foot_task = FrameTask(
        "r_ankle", position_cost=5.0, orientation_cost=5.0
    )
    pelvis_task = FrameTask(
        "PELVIS_S", position_cost=1.0, orientation_cost=1.0
    )
    right_wrist_task = FrameTask(
        "r_wrist", position_cost=1.0, orientation_cost=1.0
    )
    damping_task = DampingTask(
        cost=1e-1,
    )
    posture_task = PostureTask(
        cost=1e-2,
    )
    acceleration_task = LowAccelerationTask(
        cost=1e-1,
    )

    tasks = [
        left_foot_task, 
        pelvis_task, 
        right_foot_task, 
        right_wrist_task,
        # damping_task,
        # posture_task,
        # acceleration_task,
    ]

    pelvis_pose = configuration.get_transform_frame_to_world("PELVIS_S").copy()
    # pelvis_pose.translation[0] += 0.05
    viewer["pelvis_pose"].set_transform(pelvis_pose.np)
    pelvis_task.set_target(pelvis_pose)

    # transform_l_ankle_target_to_init = pin.SE3(
    #     np.eye(3), np.array([0.1, 0.0, 0.0])
    # )
    # transform_r_ankle_target_to_init = pin.SE3(
    #     np.eye(3), np.array([-0.1, 0.0, 0.0])
    # )

    # left_foot_task.set_target(
    #     configuration.get_transform_frame_to_world("l_ankle")
    #     * transform_l_ankle_target_to_init
    # )
    # right_foot_task.set_target(
    #     configuration.get_transform_frame_to_world("r_ankle")
    #     * transform_r_ankle_target_to_init
    # )

    transform_l_ankle_target_to_world = pin.SE3(
        np.eye(3), np.array([0.1, 0.1, -0.5])
    )
    transform_r_ankle_target_to_world = pin.SE3(
        np.eye(3), np.array([0.2, -0.3, -0.5])
    )
    left_foot_task.set_target(transform_l_ankle_target_to_world)
    right_foot_task.set_target(transform_r_ankle_target_to_world)

    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("PELVIS_S")
    )

    posture_task.set_target_from_configuration(configuration)

    # print(damping_task.compute_error(configuration).shape[0])
    # print(configuration.model.nv)
    # print(configuration.tangent.eye.shape)
    # print(acceleration_task.compute_error(configuration).shape[0])
    # print(damping_task.compute_jacobian(configuration).shape[0], damping_task.compute_jacobian(configuration).shape[1])
    # root_nq, root_nv = get_root_joint_dim(configuration.model)
    # print(root_nq, root_nv)


    right_wrist_pose = WavingPose(
        configuration.get_transform_frame_to_world("r_wrist")
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    # if "proxqp" in qpsolvers.available_solvers:
    #     solver = "proxqp"
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        pelvis_target = pelvis_task.transform_target_to_world
        pelvis_target.translation[0] = -0.05
        pelvis_target.translation[2] = 0.1 * np.sin(8.0 * t)
        viewer["pelvis_pose"].set_transform(pelvis_target.np)

        right_wrist_task.set_target(right_wrist_pose.at(t))
        wrist_frame.set_transform(right_wrist_pose.at(t).np)

        left_foot_target = left_foot_task.transform_target_to_world
        right_foot_target = right_foot_task.transform_target_to_world
        left_foot_target.translation[0] = 0.2 * np.sin(8.0 * t)
        left_foot_target.translation[2] = val if (val := 0.5 * np.sin(8.0 * t)) < -0.3 else left_foot_target.translation[2]
        viewer["left_foot_target_pose"].set_transform(left_foot_target.np)

        right_foot_target.translation[0] = 0.5
        right_foot_target.translation[1] = -0.3
        right_foot_target.translation[2] = -0.4
        viewer["right_foot_target_pose"].set_transform(right_foot_target.np)

        start_time = time.time()
        # Compute velocity and integrate it into next configuration
        # kwargs = {}
        # kwargs["max_iter"] = 100
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        # velocity = solve_ik(configuration, tasks, dt, solver=solver, **kwargs)
        end_time = time.time()
        print(f"time elapsed: {end_time - start_time:.6f} s")
        
        configuration.integrate_inplace(velocity, dt)

        acceleration_task.set_last_integration(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
