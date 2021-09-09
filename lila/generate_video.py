import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gym_ignition
import gym_ignition_environments
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import skbot.ignition as ign
import skbot.trajectory as rtj
import skbot.transform as rtf
from skbot.ignition.sdformat.bindings import v18
from gym_ignition.context.gazebo import controllers
from gym_ignition.rbd import conversions
from matplotlib.patches import Circle

# from panda_controller import LinearJointSpacePlanner
from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from scipy.spatial.transform import Rotation as R
from skbot.trajectory import spline_trajectory

# from simulator import LegibilitySimulator


class Panda(gym_ignition_environments.models.panda.Panda):
    def __init__(self, **kwargs):
        self.home_position = np.array(
            (0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.03, 0.03)
        )
        super().__init__(**kwargs)

        # Constraints

        # joint constraints (units in rad, e.g. rad/s for velocity)
        # TODO: check the values of the fingers, these are all guesses
        self.max_position = np.array(
            (2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973, 0.045, 0.045)
        )
        self.min_position = np.array(
            (
                -2.8973,
                -1.7628,
                -2.8973,
                -3.0718,
                -2.8973,
                -0.0175,
                -2.8973,
                -0.001,
                -0.001,
            )
        )
        self.max_velocity = np.array(
            (2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 0.3, 0.3)
        )
        self.min_velocity = -self.max_velocity
        self.max_acceleration = np.array(
            (15, 7.5, 10, 12.5, 15, 20, 20, 10, 10), dtype=np.float_
        )
        self.min_acceleration = -self.max_acceleration
        self.max_jerk = np.array(
            (7500, 3750, 5000, 6250, 7500, 10000, 10000, 10000, 10000), dtype=np.float_
        )
        self.min_jerk = -self.max_jerk
        self.max_torque = np.array(
            (87, 87, 87, 87, 12, 12, 12, 12, 12), dtype=np.float_
        )
        self.min_torque = -self.max_torque
        self.max_rotatum = np.array([1000] * 9)
        self.min_rotatum = -self.max_rotatum

        # tool constraints
        self.max_tool_velocity = 1.7  # m/s
        self.max_tool_acceleration = 13  # m/s
        self.max_tool_jerk = 6500  # m/s
        self.max_tool_angular_velocity = 2.5  # rad/s
        self.max_tool_angular_acceleration = 25  # rad/s
        self.max_tool_angular_jerk = 12500  # rad/s

        # ellbow constraints (in rad)
        # This is in the docs, but I'm not sure how to interpret it. Perhaps it
        # refers to null-space motion?
        # https://frankaemika.github.io/docs/control_parameters.html
        self.max_ellbow_velocity = 2.175
        self.max_ellbow_acceleration = 10
        self.max_ellbow_jerk = 5000

        panda = self.model

        panda.to_gazebo().enable_self_collisions(True)

        # Insert the ComputedTorqueFixedBase controller
        assert panda.to_gazebo().insert_model_plugin(
            *controllers.ComputedTorqueFixedBase(
                kp=[100.0] * (self.dofs - 2) + [10000.0] * 2,
                ki=[0.0] * self.dofs,
                kd=[17.5] * (self.dofs - 2) + [100.0] * 2,
                urdf=self.get_model_file(),
                joints=self.joint_names(),
            ).args()
        )

    def reset(self):
        self.position = self.home_position
        self.velocity = [0] * 9
        self.target_position = self.home_position
        self.target_velocity = [0] * 9
        self.target_acceleration = [0] * 9

    @property
    def dofs(self):
        return self.model.dofs()

    @property
    def position(self):
        return np.array(self.model.joint_positions())

    @property
    def velocity(self):
        return np.array(self.model.joint_velocities())

    @property
    def acceleration(self):
        return np.array(self.model.joint_accelerations())

    @position.setter
    def position(self, position: npt.ArrayLike):
        position = np.asarray(position)

        if np.any((position < self.min_position) | (self.max_position < position)):
            raise ValueError("The position exceeds the robot's limits.")

        assert self.model.to_gazebo().reset_joint_positions(position.tolist())

    @velocity.setter
    def velocity(self, velocity: npt.ArrayLike):
        velocity = np.asarray(velocity)

        if np.any((velocity < self.min_velocity) | (self.max_velocity < velocity)):
            raise ValueError("The velocity exceeds the robot's limits.")

        assert self.model.to_gazebo().reset_joint_velocities(velocity.tolist())

    @property
    def target_position(self):
        return np.array(self.model.joint_position_targets())

    @property
    def target_velocity(self):
        return np.array(self.model.joint_velocity_targets())

    @property
    def target_acceleration(self):
        return np.array(self.model.joint_acceleration_targets())

    @target_position.setter
    def target_position(self, position: npt.ArrayLike):
        position = np.asarray(position)

        if np.any((position < self.min_position) | (self.max_position < position)):
            raise ValueError("The target position exceeds the robot's limits.")

        assert self.model.set_joint_position_targets(position.tolist())

    @target_velocity.setter
    def target_velocity(self, velocity: npt.ArrayLike):
        velocity = np.asarray(velocity)

        if np.any((velocity < self.min_velocity) | (self.max_velocity < velocity)):
            raise ValueError("The target velocity exceeds the robot's limits.")

        assert self.model.set_joint_velocity_targets(velocity.tolist())

    @target_acceleration.setter
    def target_acceleration(self, acceleration: npt.ArrayLike):
        acceleration = np.asarray(acceleration)

        if np.any(
            (acceleration < self.min_acceleration)
            | (self.max_acceleration < acceleration)
        ):
            raise ValueError("The target acceleration exceeds the robot's limits.")

        assert self.model.set_joint_acceleration_targets(acceleration.tolist())

    @property
    def tool(self):
        return self.model.get_link("end_effector_frame")

    @property
    def tool_pose(self):
        position = self.model.get_link("end_effector_frame").position()
        orientation = self.model.get_link("end_effector_frame").orientation()

        return (position, orientation)

    @tool_pose.setter
    def tool_pose(self, pose):
        """Set the joints so that the tool is in the desired configuration"""

        position, orientation = pose
        if position is None and orientation is None:
            return

        self.position = self.solve_ik(position=position, orientation=orientation)

    # @target_tool_pose.setter
    def target_tool_pose(self, pose):
        position, orientation = pose
        self.target_position = self.solve_ik(position=position, orientation=orientation)

    @property
    def tool_velocity(self):
        return self.model.get_link("end_effector_frame").world_linear_velocity()

    @property
    def tool_angular_velocity(self):
        return self.model.get_link("end_effector_frame").world_angular_velocity()

    @property
    def tool_acceleration(self):
        return self.model.get_link("end_effector_frame").world_linear_acceleration()

    @property
    def tool_angular_acceleration(self):
        return self.model.get_link("end_effector_frame").world_angular_acceleration()


@dataclass
class ImageMessage:
    image: np.array
    time: float


def camera_parser(msg):
    image_msg = ign.messages.Image()
    image_msg.parse(msg[2])

    im = np.frombuffer(image_msg.data, dtype=np.uint8)
    im = im.reshape((image_msg.height, image_msg.width, 3))

    img_time = image_msg.header.stamp.sec + image_msg.header.stamp.nsec * 1e-9

    return ImageMessage(image=im, time=img_time)


def generate_video(trajectory: np.ndarray, environment: Path):
    env = environment
    sdf_string = env.read_text()

    world: v18.World = ign.sdformat.loads(sdf_string).world[0]

    world_frame = ign.sdformat.to_frame_graph(sdf_string, shape=(100, 3))
    tool_frame = world_frame.find_frame(".../panda_link8")
    main_camera_frame = world_frame.find_frame(".../main_camera/.../pixel-space")
    side_camera_frame = world_frame.find_frame(".../sideview_camera/.../pixel-space")
    goals = [m for m in world.model if m.name.startswith("box_copy_")]
    goal_frames = [
        world_frame.find_frame(f".../{goal.name}/box_link") for goal in goals
    ]
    goal_array = np.array(
        [f.transform(((0, 0, 0),), world_frame)[0] for f in goal_frames]
    )

    # fig, ax = plt.subplots(1)
    simulator = scenario_gazebo.GazeboSimulator(
        step_size=0.001, steps_per_run=round((1 / 0.001) / 30)
    )
    simulator.insert_world_from_sdf(str(env))

    # panda.to_gazebo().enable_self_collisions(True)

    # # Insert the ComputedTorqueFixedBase controller
    # assert panda.to_gazebo().insert_model_plugin(
    #     *controllers.ComputedTorqueFixedBase(
    #         kp=[100.0] * (self.dofs - 2) + [10000.0] * 2,
    #         ki=[0.0] * self.dofs,
    #         kd=[17.5] * (self.dofs - 2) + [100.0] * 2,
    #         urdf=self.get_model_file(),
    #         joints=self.joint_names(),
    #     ).args()
    # )


    simulator.initialize()
    # goal_px = simulator.in_px_coordinates(simulator.cubes[goal_idx].base_position())
    # ax.add_patch(Circle(goal_px, radius=10, color="red"))

    # uncomment to show the GUI
    simulator.gui()
    # simulator.prepare_goal_trajectory(goal_idx, via_point_idx=trajectory_row.iloc[0]["viaPointIdx"])
    with ign.Subscriber(
        "/main_camera", parser=camera_parser
    ) as camera_topic, ign.Subscriber(
        "/sideview_camera", parser=camera_parser
    ) as side_camera_topic:
        simulator.run(paused=True)
        for sim_step in range(330):
            # ax.add_patch(Circle(eff_px, radius=5))
            simulator.run()
        
        img_msg = camera_topic.recv()
        iio.imwrite("front_view.png", img_msg.image)
        side_img_msg = side_camera_topic.recv()
        iio.imwrite("side_view.png", side_img_msg.image)

        writer = iio.get_writer("test.mp4", format="FFMPEG", mode="I", fps=30)
        while True:
            try:
                img_msg = camera_topic.recv()
                writer.append_data(img_msg.image)
            except:
                break
        writer.close()

        writer = iio.get_writer("test_side.mp4", format="FFMPEG", mode="I", fps=30)
        while True:
            try:
                side_img_msg = side_camera_topic.recv()
                writer.append_data(side_img_msg.image)
            except:
                break
        writer.close()


    # # visualize the trajectory
    # ax.imshow(img_msg.image)
    # ax.set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # fig.savefig(trajectory_row.iloc[0]["imageFile"])


if __name__ == "__main__":
    generate_video(
        np.zeros((100, 9), dtype=np.float_),
        Path(__file__).parent / "sdf" / "four_goals.sdf",
    )
