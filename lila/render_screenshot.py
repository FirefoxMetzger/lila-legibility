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

    root_frame = ign.sdformat.to_frame_graph(sdf_string)
    px_space = root_frame.find_frame(".../pixel-space")
    cam_space = root_frame.find_frame(".../camera-space")
    box = root_frame.find_frame(".../box_visual")
    vertices = np.array(
        [
            [0.025, 0.025, 0.025],#0
            [0.025, -0.025, 0.025],
            [0.025, 0.025, -0.025],
            [0.025, -0.025, -0.025],
            [-0.025, 0.025, 0.025],
            [-0.025, 0.025, -0.025],#5
            [-0.025, -0.025, 0.025],
            [-0.025, -0.025, -0.025],
        ]
    )

    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 6),
        (2, 3),
        (2, 5),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7)
    ]


    simulator = scenario_gazebo.GazeboSimulator(
        step_size=0.001, steps_per_run=round((1 / 0.001) / 30)
    )
    simulator.insert_world_from_sdf(str(env))
    simulator.initialize()

    with ign.Subscriber(
        "/main_camera", parser=camera_parser
    ) as camera_topic:
        simulator.run(paused=True)
        simulator.run()
        
        img_msg = camera_topic.recv()

    iio.imwrite("test_image.png", img_msg.image)

    _, ax = plt.subplots(1)
    ax.imshow(img_msg.image)
    
    corner_px = box.transform(vertices, px_space)
    ax.scatter(corner_px[:, 1], corner_px[:, 0], 1)

    distance = list()
    for idx_a, idx_b in edges:
        x = np.linspace(corner_px[idx_a, 1], corner_px[idx_b, 1], 100)
        y = np.linspace(corner_px[idx_a, 0], corner_px[idx_b, 0], 100)
        ax.plot(x, y, "red")
        distance.append(np.linalg.norm(y - x))

    center = np.array([0,0,0])
    ax.add_patch(Circle(box.transform(center, px_space)[::-1], radius=1, color="red"))
    print(f"Center pos: {box.transform(center, px_space)}")
    print(f"Expected: {simulator.get_world().get_model('camera').get_link('link').position()}")
    print(f"Actual: {root_frame.find_frame('.../camera/link').transform(center, root_frame)}")

    plt.show()


if __name__ == "__main__":
    generate_video(
        np.zeros((100, 9), dtype=np.float_),
        Path(__file__).parent / "sdf" / "perspective_transform.sdf",
    )
