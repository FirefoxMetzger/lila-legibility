from skbot.ignition.sdformat.bindings import v18
import skbot.transform as tf
import skbot.ignition as ign
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from typing import List
import skbot.inverse_kinematics as ik
from scipy.optimize import minimize
import skbot.trajectory as trj


def plan_trajectory(environment:Path, metric):
    sdf_string = environment.read_text()
    world:v18.World = ign.sdformat.loads(sdf_string).world[0]

    world_frame = ign.sdformat.to_frame_graph(sdf_string, shape=(100, 3))
    tool_frame = world_frame.find_frame(".../panda_link8")
    main_camera_frame = world_frame.find_frame(".../main_camera/.../pixel-space")
    side_camera_frame = world_frame.find_frame(".../sideview_camera/.../pixel-space")
    goals = [m for m in world.model if m.name.startswith("box_copy_")]
    goal_frames = [world_frame.find_frame(f".../{goal.name}/box_link") for goal in goals]
    goal_array = np.array([f.transform(((0,0,0),), world_frame)[0] for f in goal_frames])

    joints = list()
    for link in tool_frame.transform_chain(world_frame):
        if isinstance(link, (tf.RotationalJoint, tf.PrismaticJoint)):
            joints.append(link)

    # initial position is home position
    trajectory = np.copy(np.broadcast_to((0, -0.785,0, -2.356, 0, 1.571, 0.785, 0.03, 0.03), (100, 9)), order="C")
    for arr, joint in zip(np.split(trajectory, 9, axis=1),joints):
            joint.param = arr
    
    planning_space = world_frame
    planning_dim = planning_space.ndim
    planning_goals = world_frame.transform(goal_array, planning_space)
    planning_trajectory = tool_frame.transform(((0,0,0),), planning_space)
    goal_idx = 0
    
    result = minimize(lambda x: metric(x.reshape(100, planning_dim), goal_array)[goal_idx], planning_trajectory)
    optimal_trajectory = np.empty((101, planning_space.ndim), dtype=np.float_)
    optimal_trajectory[:-1, ...] = result.x.reshape(100, planning_dim)
    optimal_trajectory[-1, ...] = planning_goals[goal_idx]
    optimal_trajectory = trj.spline_trajectory(np.linspace(0, 1, 100), optimal_trajectory)

    trajectory_joint_space = np.empty((100, 9), dtype=np.float_)
    for idx in reversed(range(100)):
        world_pos = optimal_trajectory[idx, ...]
        joint_pos = ik.ccd((0,0,0), world_pos, tool_frame, world_frame, joints)
        trajectory_joint_space[idx, ...] = joint_pos + [0.03, 0.03]

    print("")


def cost(trajectory:ArrayLike, goals:List[ArrayLike]) -> List[float]:
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

    trajectory_length = np.sum(np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=-1))
    return [trajectory_length + np.linalg.norm(goal - trajectory[-1]) for goal in goals]



if __name__ == "__main__":
    plan_trajectory(Path(__file__).parent / "sdf" / "four_goals.sdf", cost)