import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import generators
import gym_ignition
import gym_ignition_environments
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import skbot.ignition as ign
import skbot.trajectory as rtj
import skbot.transform as rtf
from gym_ignition.context.gazebo import controllers
from gym_ignition.rbd import conversions
from matplotlib.patches import Circle
from panda_controller import LinearJointSpacePlanner
from scenario import core as scenario_core
from scenario import gazebo as scenario_gazebo
from scipy.spatial.transform import Rotation as R
from skbot.trajectory import spline_trajectory

from simulator import LegibilitySimulator


class Panda(LinearJointSpacePlanner, gym_ignition_environments.models.panda.Panda):
    def __init__(self, **kwargs):
        self.home_position = np.array((0, -0.785,0, -2.356, 0, 1.571, 0.785, 0.03, 0.03))
        super().__init__(**kwargs)

        # Constraints

        # joint constraints (units in rad, e.g. rad/s for velocity)
        # TODO: check the values of the fingers, these are all guesses
        self.max_position = np.array((2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973, 0.045, 0.045))
        self.min_position = np.array((-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.001, -0.001))
        self.max_velocity = np.array((2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 0.3, 0.3))
        self.min_velocity = - self.max_velocity
        self.max_acceleration = np.array((15, 7.5, 10, 12.5, 15, 20, 20, 10, 10), dtype=np.float_)
        self.min_acceleration = - self.max_acceleration
        self.max_jerk = np.array((7500, 3750, 5000, 6250, 7500, 10000, 10000, 10000, 10000), dtype=np.float_)
        self.min_jerk = - self.max_jerk
        self.max_torque = np.array((87, 87, 87, 87, 12, 12, 12, 12, 12), dtype=np.float_)
        self.min_torque = - self.max_torque
        self.max_rotatum = np.array([1000] * 9)
        self.min_rotatum = - self.max_rotatum

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

        if np.any((acceleration < self.min_acceleration) | (self.max_acceleration < acceleration)):
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

    def solve_ik(self, *, position=None, orientation=None):
        if position is None and orientation is None:
            return self.position

        old_position, old_orientation = self.tool_pose
        position = old_position if position is None else position
        orientation = old_orientation if orientation is None else orientation

        # reset IK
        self.ik.set_current_robot_configuration(
            base_position=np.array(self.base_position()),
            base_quaternion=np.array(self.base_orientation()),
            joint_configuration=self.home_position,
        )
        self.ik.solve()

        return super().solve_ik(position, orientation)

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

class PandaMixin:
    """Add a Panda Robot to the simulator"""

    def __init__(self, *, panda_config, **kwargs):
        super().__init__(**kwargs)

        self.panda = None
        self.config = panda_config

    def initialize(self):
        super().initialize()
        self.config["world"] = self.world
        self.panda = Panda(**self.config)
        panda = self.panda
        assert panda.set_controller_period(period=self.step_size())

        panda.reset()
        super().run(paused=True)  # update the controller positions
        panda.target_position = panda.position



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

class ModelSpawnerMixin:
    """Simulator Mixin to spawn objects"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cache = dict()

    def insert_model(
        self,
        model_template: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        velocity: Tuple[float, float, float] = (0, 0, 0),
        angular_velocity: Tuple[float, float, float] = (0, 0, 0),
        *,
        name_prefix="",
        **template_parameters
    ):
        """Spawn the model into the simulation world"""

        if model_template not in self.model_cache:
            with open(model_template, "r") as file:
                self.model_cache[model_template] = file.read()

        if isinstance(velocity, np.ndarray):
            velocity = velocity.tolist()

        if isinstance(angular_velocity, np.ndarray):
            angular_velocity = angular_velocity.tolist()

        world = self.world

        model = self.model_cache[model_template].format(**template_parameters)
        model_name = gym_ignition.utils.scenario.get_unique_model_name(
            world=world, model_name=name_prefix
        )
        pose = scenario_core.Pose(position, orientation)
        assert world.insert_model(model, pose, model_name)

        obj = world.get_model(model_name)

        velocity = scenario_core.Array3d(velocity)
        assert obj.to_gazebo().reset_base_world_linear_velocity(velocity)

        angular_velocity = scenario_core.Array3d(angular_velocity)
        assert obj.to_gazebo().reset_base_world_angular_velocity(velocity)

        return obj


class CameraMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.camera = None

        # --- initialize camera projection ---
        camera_frequency = 30  # Hz
        self.steps_per_frame = round((1 / self.step_size()) / camera_frequency)
        self.runs_per_frame = round(self.steps_per_run() / self.steps_per_frame)

        self.cam_intrinsic = rtf.perspective_frustum(
            hfov=1.13446, image_shape=(1080, 1920)
        )
        self.cam_extrinsic = None

    def initialize(self):
        super().initialize()
        self.camera = self.get_world().get_model("camera").get_link("link")

        # extrinsic matrix
        camera = self.camera
        cam_pos_world = np.array(camera.position())
        cam_ori_world_quat = np.array(camera.orientation())[[1, 2, 3, 0]]
        cam_ori_world = R.from_quat(cam_ori_world_quat).as_euler("xyz")
        camera_frame_world = np.stack((cam_pos_world, cam_ori_world)).ravel()
        self.cam_extrinsic = rtf.transform(camera_frame_world)

    def in_px_coordinates(self, position):
        pos_world = rtf.homogenize(position)
        pos_cam = np.matmul(self.cam_extrinsic, pos_world)
        pos_px_hom = np.matmul(self.cam_intrinsic, pos_cam)
        pos_px = rtf.cartesianize(pos_px_hom)

        return pos_px


class LegibilitySimulator(
    ModelSpawnerMixin, CameraMixin, PandaMixin, scenario_gazebo.GazeboSimulator
):
    def __init__(self, environment: generators.Environment, **kwargs):
        super().__init__(**kwargs)

        self.env = environment
        self.cubes = list()

        self.path = None
        self.path_velocity = None
        self.path_time_start = 0
        self.path_time_end = 10


    @property
    def world(self):
        return self.get_world()

    def initialize(self):
        raise NotImplementedError(
            "Do not initialize the simulator manually. "
            "Use 'with LegibiltySimulator(...) as' instead."
        )

    def __enter__(self):
        assert self.insert_world_from_sdf("./sdf/environment.sdf")
        super().initialize()
        return self

    def prepare_goal_trajectory(self, cube_idx, via_point_idx=None):
        cube = self.cubes[cube_idx]
        cube_position = np.array(cube.base_position())
        ori = R.from_quat(np.array(cube.base_orientation())[[1, 2, 3, 0]])

        tool_pos, tool_rot = self.panda.tool_pose

        if via_point_idx is None:
            idx = random.randint(0, len(self.env.control_points))
        else:
            idx = via_point_idx
        via_point = self.env.control_points[idx] + self.panda.base_position()

        # key in the movement's poses
        #   0 - home_position
        #   1 - random via-point
        #   2 - above cube
        #   3 - at cube (open gripper)
        #   4 - grabbing cube (closed gripper)
        #   5 - home_position (holding cube)
        pose_keys = np.empty((6, 9), dtype=np.float_)
        pose_keys[0] = self.panda.home_position
        pose_keys[1] = self.panda.solve_ik(position=via_point)
        pose_keys[2] = self.panda.solve_ik(position=(cube_position + np.array((0, 0, 0.01))))
        pose_keys[2, -2:] = (.04, .04)
        pose_keys[3] = self.panda.solve_ik(position=cube_position)
        pose_keys[3, -2:] = (.04, .04)
        pose_keys[4] = self.panda.solve_ik(position = cube_position)
        pose_keys[4, -2:] = (0, 0)
        pose_keys[5] = self.panda.home_position
        pose_keys[5, -2:] = (0, 0)


        # set keyframe times
        trajectory_duration = self.path_time_end - self.path_time_start
        times = np.array([0, 0.275, 0.55, 0.6, 0.7, 1]) * trajectory_duration + self.path_time_start

        if self.path_time_start % self.step_size() != 0:
            raise RuntimeError("Path does not start during a simulator step.")

        t = np.arange(
            self.path_time_start+self.step_size(),
            self.path_time_end+self.step_size(),
            self.step_size()
        )
        self.path = spline_trajectory(
            t, pose_keys, t_control=times, degree=1
        )
        self.path_velocity = spline_trajectory(
            t,
            pose_keys,
            t_control=times,
            degree=1,
            derivative=1,
        )


    def run(self, **kwargs):
        sim_time = self.world.time()
        if self.path_time_start < sim_time < self.path_time_end:
            idx = int(round((sim_time - self.path_time_start) / self.step_size()))
            self.panda.target_position = self.path[idx]
            self.panda.target_velocity = self.path_velocity[idx]

        super().run(**kwargs)


    def __exit__(self, type, value, traceback):
        assert self.close()


def generate_video(trajectory, environment):
    env_idx = int(sys.argv[1])
    goal_idx = int(sys.argv[2])
    trajectory_idx = int(sys.argv[3])
    random.seed(datetime.now())

    dataset_root = Path(__file__).parents[0] / "dataset"
    env_meta = pd.read_excel(dataset_root / "environment_metadata.xlsx").set_index("Unnamed: 0")
    trajectory_meta = pd.read_excel(dataset_root / "trajectory_metadata.xlsx").set_index("Unnamed: 0")
    env_root = Path(".") / env_meta.loc[env_idx].DataDir

    with open(env_meta.loc[env_idx].EnvironmentFilePath, "rb") as file:
        env = pickle.load(file)

    # execute the trajectory and record
    # - endeffector position in planning space

    cam_trajectory = list()
    joint_trajectory = list()
    world_trajectory = list()
    writer = iio.get_writer(trajectory_row.iloc[0]["videoFile"], format="FFMPEG", mode="I", fps=30)
    fig, ax = plt.subplots(1)
    panda_config = {"position": [0.2, 0, 1.025]}
    with LegibilitySimulator(
        panda_config=panda_config,
        environment=env,
        step_size=0.001,
        rtf=1.0,
        steps_per_run=round((1 / 0.001) / 30),
    ) as simulator:
        goal_px = simulator.in_px_coordinates(simulator.cubes[goal_idx].base_position())
        ax.add_patch(Circle(goal_px, radius=10, color="red"))

        cubes_world = np.stack([cube.base_position() for cube in simulator.cubes])
        np.save(env_root / "world_cube_position.npy", cubes_world)
        cubes_cam = np.stack([simulator.in_px_coordinates(pos) for pos in cubes_world])
        np.save(env_root / "camera_cube_position.npy", cubes_cam)
        cubes_joint = np.stack([simulator.panda.solve_ik(position=pos) for pos in cubes_world])
        np.save(env_root / "joint_cube_position.npy", cubes_joint)

        # uncomment to show the GUI
        # simulator.gui()
        simulator.prepare_goal_trajectory(goal_idx, via_point_idx=trajectory_row.iloc[0]["viaPointIdx"])
        with ign.Subscriber("/camera", parser=camera_parser) as camera_topic:
            simulator.run(paused=True)
            for sim_step in range(330):
                img_msg = camera_topic.recv()
                writer.append_data(img_msg.image)

                eff_px = simulator.in_px_coordinates(simulator.panda.tool_pose[0])

                cam_trajectory.append(eff_px)
                joint_trajectory.append(simulator.panda.position)
                world_trajectory.append(simulator.panda.tool_pose[0])

                ax.add_patch(Circle(eff_px, radius=5))
                simulator.run()

    writer.close()

    # visualize the trajectory
    ax.imshow(img_msg.image)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(trajectory_row.iloc[0]["imageFile"])

    trajectory_meta = trajectory_meta.append(trajectory_row)
    trajectory_meta.to_excel(dataset_root / "trajectory_metadata.xlsx")
