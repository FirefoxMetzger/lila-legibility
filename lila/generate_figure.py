import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import gym_ignition
import gym_ignition_environments
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import skbot.ignition as ign
import skbot.trajectory as rtj
import skbot.transform as tf
from skbot.ignition.sdformat.bindings import v18
from gym_ignition.context.gazebo import controllers
from gym_ignition.rbd import conversions
from matplotlib.patches import Circle
import gym_ignition_models
import tempfile
import time

from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from scipy.spatial.transform import Rotation as R
from skbot.trajectory import spline_trajectory
import skbot.inverse_kinematics as ik
from skimage.draw import circle_perimeter as skimage_circle


def generate_figure(trajectory: Path, environment: Path, out_filename: str):
    current_idx = 0
    trajectory_array = np.load(trajectory)
    t_control = trajectory_array.get("time")
    angles = trajectory_array.get("joint_position")
    angular_velocity = trajectory_array.get("joint_velocity")
    world_pos = trajectory_array.get("world_position")

    env = environment
    sdf_string = env.read_text()

    generic_sdf = ign.sdformat.loads_generic(sdf_string)

    frames = generic_sdf.worlds[0].declared_frames()
    generic_sdf.worlds[0].to_dynamic_graph(frames)

    world_frame = frames["world"]
    tool_frame = frames["panda::panda_link8"]
    base_frame = frames["panda::panda_link0"]
    main_camera_frame = frames["main_camera::link::camera::pixel_space"]
    angle_camera_frame = frames["angle_camera::link::camera::pixel_space"]
    side_camera_frame = frames["side_camera::link::camera::pixel_space"]
    num_goals = len(
        [m for m in generic_sdf.worlds[0].models if m.name.startswith("box_copy_")]
    )
    goal_frames = [frames[f"box_copy_{idx}::box_link"] for idx in range(num_goals)]
    goal_array = np.array(
        [f.transform((0, 0, 0), frames["world"])[0] for f in goal_frames]
    )

    joint_list = [
        x
        for x in tool_frame.transform_chain(goal_frames[0])
        if isinstance(x, tf.RotationalJoint)
    ]
    joint_list = [x for x in reversed(joint_list)]

    # step_order = [5, 4, 3, 6, 2, 1, 0]
    step_order = [6, 5, 4, 3, 2, 1, 0]
    joint_list = [joint_list[idx] for idx in step_order]

    # skbot_joint_links = [x for x in tool_frame.transform_chain(goal_frames[0]) if isinstance(x, tf.RotationalJoint)]

    sdf_obj = ign.sdformat.loads(sdf_string)
    for model in sdf_obj.world[0].include:
        if model.name == "panda":
            sdf_obj.world[0].include.remove(model)
            break
    new_sdf_string = ign.sdformat.dumps(sdf_obj)
    simulator = scenario_gazebo.GazeboSimulator(
        step_size=0.001, steps_per_run=round((1 / 0.001) / 30)
    )

    with tempfile.NamedTemporaryFile(mode="r+") as f:
        f.write(new_sdf_string)
        f.seek(0)
        assert simulator.insert_world_from_sdf(f.name)

    assert simulator.initialize()
    world = simulator.get_world()
    assert world.set_physics_engine(scenario_gazebo.PhysicsEngine_dart)
    # goal_px = simulator.in_px_coordinates(simulator.cubes[goal_idx].base_position())
    # ax.add_patch(Circle(goal_px, radius=10, color="red"))

    panda = gym_ignition_environments.models.panda.Panda(
        world, position=[0.2, 0.0, 1.025]
    )
    panda.to_gazebo().enable_self_collisions(True)
    # joints = [name for name in panda.joint_names() if "panda_joint" in name]

    # Set the controller period
    assert panda.set_controller_period(period=simulator.step_size())

    # Insert the ComputedTorqueFixedBase controller
    assert panda.to_gazebo().insert_model_plugin(
        *controllers.ComputedTorqueFixedBase(
            kp=[100.0] * (panda.dofs() - 2) + [10000.0] * 2,
            ki=[0.0] * panda.dofs(),
            kd=[17.5] * (panda.dofs() - 2) + [100.0] * 2,
            urdf=panda.get_model_file(),
            joints=panda.joint_names(),
        ).args()
    )

    simulator.run(paused=True)

    # sync state
    for idx, link in enumerate(reversed(joint_list)):
        link.param = panda.joint_positions()[idx]

    assert panda.set_joint_position_targets(panda.joint_positions())
    assert panda.set_joint_velocity_targets(panda.joint_velocities())
    assert panda.set_joint_acceleration_targets(panda.joint_accelerations())

    # uncomment to show the GUI
    simulator.gui()
    simulator.run(paused=True)
    time.sleep(3)
    with ign.Subscriber(
        "/main_camera"
    ) as camera_topic, ign.Subscriber(
        "/side_camera"
    ) as side_camera_topic, ign.Subscriber(
        "/angle_camera"
    ) as angle_camera_topic:
        simulator.run()

        front_msg = camera_topic.recv()
        shape = (front_msg.height, front_msg.width, 3)
        front_image = np.frombuffer(front_msg.data, dtype=np.uint8).reshape(shape)

        angle_camera_msg = angle_camera_topic.recv()
        angle_image = np.frombuffer(angle_camera_msg.data, dtype=np.uint8).reshape(
            shape
        )

        side_img_msg = side_camera_topic.recv()
        side_image = np.frombuffer(side_img_msg.data, dtype=np.uint8).reshape(shape)

    fig_front, ax_front = plt.subplots()
    ax_front.set_axis_off()
    ax_front.imshow(front_image)
    planned_trajectory = world_frame.transform(world_pos, main_camera_frame)
    ax_front.scatter(*planned_trajectory[:, [1, 0]].T, s=2, c="tab:red")

    fig_angle, ax_angle = plt.subplots()
    ax_angle.set_axis_off()
    ax_angle.imshow(angle_image)
    planned_trajectory = world_frame.transform(world_pos, angle_camera_frame)
    ax_angle.scatter(*planned_trajectory[:, [1, 0]].T, s=2, c="tab:red")

    fig_side, ax_side = plt.subplots()
    ax_side.set_axis_off()
    ax_side.imshow(side_image)
    planned_trajectory = world_frame.transform(world_pos, side_camera_frame)
    ax_side.scatter(*planned_trajectory[:, [1, 0]].T, s=2, c="tab:red")


    front_trajectory = list()
    angle_trajectory = list()
    side_trajectory = list()
    while current_idx < len(angles):
        simulator.run()

        # sync state
        for idx, link in enumerate(joint_list):
            joint_idx_sim = step_order[idx]
            link.param = panda.joint_positions()[joint_idx_sim]

        front_trajectory.append(tool_frame.transform((0,0,0), main_camera_frame))
        angle_trajectory.append(tool_frame.transform((0,0,0), angle_camera_frame))
        side_trajectory.append(tool_frame.transform((0,0,0), side_camera_frame))
        
        if world.time() > t_control[current_idx]:
            current_idx += 1
            if current_idx == len(angles):
                break

            assert panda.set_joint_position_targets(angles[current_idx])
            assert panda.set_joint_velocity_targets(angular_velocity[current_idx])

    ax_front.scatter(*np.stack(front_trajectory)[:, [1, 0]].T, s=2, c="tab:blue")
    ax_angle.scatter(*np.stack(angle_trajectory)[:, [1, 0]].T, s=2, c="tab:blue")
    ax_side.scatter(*np.stack(side_trajectory)[:, [1, 0]].T, s=2, c="tab:blue")


    fig_front.savefig(
        Path(__file__).parents[1] / "images" / (out_filename + "_front.png")
    )
    fig_angle.savefig(
        Path(__file__).parents[1] / "images" / (out_filename + "_angle.png")
    )
    fig_side.savefig(Path(__file__).parents[1] / "images" / (out_filename + "_side.png"))


if __name__ == "__main__":
    # generate_figure(
    #     Path(__file__).parents[1] / "trajectories" / "test.npz",
    #     Path(__file__).parent / "sdf" / "four_goals.sdf",
    #     "test",
    # )

    generate_figure(
        Path(__file__).parents[1] / "trajectories" / "test_arch.npz",
        Path(__file__).parent / "sdf" / "four_goals.sdf",
        "test_arch",
    )
