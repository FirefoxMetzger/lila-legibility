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


def generate_video(trajectory: np.ndarray, environment: Path, out_filename: str):
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

    panda = gym_ignition_environments.models.panda.Panda(
        world, position=[0.2, 0.0, 1.025]
    )
    panda.to_gazebo().enable_self_collisions(True)

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
        while world.time() < 10:
            simulator.run()

            # sync state
            for idx, link in enumerate(joint_list):
                joint_idx_sim = step_order[idx]
                link.param = panda.joint_positions()[joint_idx_sim]

            if world.time() > t_control[current_idx]:
                current_idx = np.argmin(np.abs(t_control - world.time()))

                assert panda.set_joint_position_targets(angles[current_idx])
                assert panda.set_joint_velocity_targets(angular_velocity[current_idx])
            
        
        out_file = Path(__file__).parents[1] / "videos" / (out_filename+"_front.mp4")
        writer = iio.get_writer(out_file, format="FFMPEG", mode="I", fps=30)
        while True:
            try:
                front_msg = camera_topic.recv()
                shape = (front_msg.height, front_msg.width, 3)
                front_image = np.frombuffer(front_msg.data, dtype=np.uint8).reshape(shape)
                writer.append_data(front_image)
            except:
                break
        writer.close()

        out_file = Path(__file__).parents[1] / "videos" / (out_filename+"_side.mp4")
        writer = iio.get_writer(out_file, format="FFMPEG", mode="I", fps=30)
        while True:
            try:
                side_img_msg = side_camera_topic.recv()
                side_image = np.frombuffer(side_img_msg.data, dtype=np.uint8).reshape(shape)
                writer.append_data(side_image)
            except:
                break
        writer.close()

        out_file = Path(__file__).parents[1] / "videos" / (out_filename+"_angle.mp4")
        writer = iio.get_writer(out_file, format="FFMPEG", mode="I", fps=30)
        while True:
            try:
                angle_camera_msg = angle_camera_topic.recv()
                angle_image = np.frombuffer(angle_camera_msg.data, dtype=np.uint8).reshape(
                    shape
                )
                writer.append_data(angle_image)
            except:
                break
        writer.close()


if __name__ == "__main__":
    # generate_video(
    #     Path(__file__).parents[1] / "trajectories" / "straight.npz",
    #     Path(__file__).parent / "sdf" / "four_goals.sdf",
    #     "straight"
    # )

    # generate_video(
    #     Path(__file__).parents[1] / "trajectories" / "arch.npz",
    #     Path(__file__).parent / "sdf" / "four_goals.sdf",
    #     "arch"
    # )

    generate_video(
        Path(__file__).parents[1] / "trajectories" / "arch_main_camera.npz",
        Path(__file__).parent / "sdf" / "four_goals.sdf",
        "arch_main_camera"
    )