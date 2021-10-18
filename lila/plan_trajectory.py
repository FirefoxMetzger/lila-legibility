import skbot.transform as tf
import skbot.ignition as ign
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from typing import List
import skbot.inverse_kinematics as ik
from scipy.optimize import minimize
import skbot.trajectory as trj


def plan_direct(environment:Path, goal_idx:int, out_file_name:str) -> ArrayLike:
    sdf_string = environment.read_text()
    generic_sdf = ign.sdformat.loads_generic(sdf_string)

    frames = generic_sdf.worlds[0].declared_frames()
    generic_sdf.worlds[0].to_dynamic_graph(frames)

    world_frame = frames["world"]
    tool_frame = frames["panda::panda_link8"]
    base_frame = frames["panda::panda_link0"]
    main_camera_frame = frames["main_camera::link::camera::pixel_space"]
    angle_camera_frame = frames["angle_camera::link::camera::pixel_space"]
    side_camera_frame = frames["side_camera::link::camera::pixel_space"]
    num_goals = len([m for m in generic_sdf.worlds[0].models if m.name.startswith("box_copy_")])
    goal_frames = [
        frames[f"box_copy_{idx}::box_link"] for idx in range(num_goals)
    ]
    goal_array = np.array(
        [f.transform((0, 0, 0), frames["world"])[0] for f in goal_frames]
    )

    joint_list = [x for x in tool_frame.transform_chain(goal_frames[0]) if isinstance(x, tf.RotationalJoint)]
    joint_list = [x for x in reversed(joint_list)]

    start = tool_frame.transform((0,0,0), world_frame)
    end = goal_frames[goal_idx].transform((0,0, 0.025), world_frame)
    t = np.linspace(0, 10, 100)
    cartesian_targets = np.linspace(start, end, 100)
    joint_targets = np.zeros((100, 9), dtype=float)

    for value, joint in zip([0, -0.785, 0, -2.356, 0, 1.571, 0.785], joint_list):
        joint.param = value

    for idx in range(100):
        pos = cartesian_targets[idx]
        targets = [
            ik.PositionTarget((0,0,0), pos, tool_frame, world_frame)
        ]
        joint_targets[idx, :7] = ik.ccd(targets, [x for x in reversed(joint_list)])[::-1]

    joint_angles = trj.spline_trajectory(t, joint_targets, t_control=t)
    joint_velocities = trj.spline_trajectory(t, joint_targets, t_control=t, derivative=1)

    np.savez(
        Path(__file__).parents[1] / "trajectories" / out_file_name,
        time = t,
        position = joint_angles,
        velocity = joint_velocities
    )


def plan_trajectory(environment: Path, metric):
    sdf_string = environment.read_text()
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

    joints = list()
    for link in tool_frame.transform_chain(world_frame):
        if isinstance(link, (tf.RotationalJoint, tf.PrismaticJoint)):
            joints.append(link)

    # initial position is home position
    trajectory = np.copy(
        np.broadcast_to((0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.03, 0.03), (100, 9)),
        order="C",
    )
    for arr, joint in zip(np.split(trajectory, 9, axis=1), joints):
        joint.param = arr

    planning_space = world_frame
    planning_dim = planning_space.ndim
    planning_goals = world_frame.transform(goal_array, planning_space)
    planning_trajectory = tool_frame.transform(((0, 0, 0),), planning_space)
    goal_idx = 0

    fig, ax = plt.subplots(1)
    front_view = iio.imread("front_view.png")
    goal_cam = np.array(
        [f.transform(((0, 0, 0),), main_camera_frame)[0] for f in goal_frames]
    )
    ax.imshow(front_view)
    for g in goal_cam:
        ax.add_patch(Circle(g[::-1], radius=10, color="red"))
    plt.show()

    result = minimize(
        lambda x: metric(x.reshape(100, planning_dim), goal_array)[goal_idx],
        planning_trajectory,
    )
    optimal_trajectory = np.empty((101, planning_space.ndim), dtype=np.float_)
    optimal_trajectory[:-1, ...] = result.x.reshape(100, planning_dim)
    optimal_trajectory[-1, ...] = planning_goals[goal_idx]
    optimal_trajectory = trj.spline_trajectory(
        np.linspace(0, 1, 100), optimal_trajectory
    )

    trajectory_joint_space = np.empty((100, 9), dtype=np.float_)
    for idx in reversed(range(100)):
        world_pos = optimal_trajectory[idx, ...]
        joint_pos = ik.ccd((0, 0, 0), world_pos, tool_frame, world_frame, joints)
        trajectory_joint_space[idx, ...] = joint_pos + [0.03, 0.03]

    return {
        "joint_space": trajectory_joint_space,
        "world_space": optimal_trajectory,
        "main_camera_space": world_frame.transform(optimal_trajectory, main_camera_frame),
        "side_camera_space": world_frame.transform(optimal_trajectory, side_camera_frame)
    }


def cost(trajectory: ArrayLike, goals: List[ArrayLike]) -> List[float]:
    """Path Length Cost

    Parameters
    ----------
    trajectory : ArrayLike
        The trajectory as a batch of control points. Shape: (batch, control_point)
    goals : List[ArrayLike]
        A list of goals in which the movement may end.

    Returns
    -------
    costs : List[float]
        A list of legibility scores for each goal.

    Notes
    -----
    The metric inserts the goal point as last element of the path.

    """

    trajectory = np.asarray(trajectory)

    trajectory_length = np.sum(
        np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=-1)
    )
    return [trajectory_length + np.linalg.norm(goal - trajectory[-1]) for goal in goals]


if __name__ == "__main__":
    import imageio as iio
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    environment = Path(__file__).parent / "sdf" / "four_goals.sdf"
    plan_direct(environment, 0, "test.npz")
    # trajectories = plan_trajectory(environment, cost)

    # fig, ax = plt.subplots(1)
    # main_cam_traj = trajectories["main_camera_space"]
    # front_view = iio.imread("front_view.png")
    # ax.imshow(front_view)
    # ax.add_patch(Circle(main_cam_traj[-1], radius=10, color="red"))

    # plt.show()