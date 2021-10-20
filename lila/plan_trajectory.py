import skbot.transform as tf
import skbot.ignition as ign
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from typing import List
import skbot.inverse_kinematics as ik
from scipy.optimize import minimize
import skbot.trajectory as trj


def plan_direct(environment:Path, goal_idx:int, out_file_name:str) -> None:
    num_plan_points = 300
    num_control_points = 100
    
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

    for value, joint in zip([0, -0.785, 0, -2.356, 0, 1.571, 0.785], joint_list):
        joint.param = value

    start = tool_frame.transform((0,0,0), world_frame)
    end = goal_frames[goal_idx].transform((0,0, 0.03), world_frame)
    t = np.linspace(0, 10, num_plan_points)
    cartesian_targets = np.linspace(start, end, num_plan_points)
    joint_targets = np.zeros((num_plan_points, 9), dtype=float)

    for idx in range(num_plan_points):
        pos = cartesian_targets[idx]
        
        try:
            targets = [
                ik.PositionTarget((0,0,0), pos, tool_frame, world_frame),
                ik.RotationTarget(tf.EulerRotation("X", -180, degrees=True), world_frame, tool_frame)
            ]
            joint_targets[idx, :7] = ik.gd(targets, joint_list)
        except RuntimeError:
            targets = [
                ik.PositionTarget((0,0,0), pos, tool_frame, world_frame),
            ]
            joint_targets[idx, :7] = ik.ccd(targets, [x for x in reversed(joint_list)])[::-1]

        for joint, value in zip(joint_list, joint_targets[idx, :7]):
            joint.angle = value

    t_final = np.linspace(0, 10, num_control_points)
    joint_angles = trj.spline_trajectory(t_final, joint_targets, t_control=t)
    joint_velocities = trj.spline_trajectory(t_final, joint_targets, t_control=t, derivative=1)

    np.savez(
        Path(__file__).parents[1] / "trajectories" / out_file_name,
        time = t_final,
        joint_position = joint_angles,
        joint_velocity = joint_velocities,
        world_position = cartesian_targets,
    )


def plan_arch(environment:Path, goal_idx:int, out_file_name:str) -> None:
    num_plan_points = 300
    num_control_points = 50
    
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

    for value, joint in zip([0, -0.785, 0, -2.356, 0, 1.571, 0.785], joint_list):
        joint.param = value

    start = tool_frame.transform((0,0,0), world_frame)
    end = goal_frames[goal_idx].transform((0,0, 0.03), world_frame)
    t = np.linspace(0, 10, num_plan_points)
    cartesian_targets = np.linspace(start, end, num_plan_points)

    goal_positions = np.stack([x.transform((0,0,0), world_frame) for x in goal_frames])
    displacement = 0.3*np.sin(np.linspace(0, np.pi, num_plan_points))

    gradients = np.sum(cartesian_targets[None, ...] - goal_positions[:, None, :], axis=0)
    gradients /= np.linalg.norm(gradients, axis=-1)[..., None]
    gradients *= displacement[..., None]
    gradients[[0,-1],:] = 0
    cartesian_targets += gradients


    joint_targets = np.zeros((num_plan_points, 9), dtype=float)
    for idx in range(num_plan_points):
        pos = cartesian_targets[idx]
        
        try:
            targets = [
                ik.PositionTarget((0,0,0), pos, tool_frame, world_frame),
                ik.RotationTarget(tf.EulerRotation("X", -180, degrees=True), world_frame, tool_frame)
            ]
            joint_targets[idx, :7] = ik.gd(targets, joint_list)
        except RuntimeError:
            targets = [
                ik.PositionTarget((0,0,0), pos, tool_frame, world_frame),
            ]
            joint_targets[idx, :7] = ik.ccd(targets, [x for x in reversed(joint_list)])[::-1]

        for joint, value in zip(joint_list, joint_targets[idx, :7]):
            joint.angle = value

    t_final = np.linspace(0, 10, num_control_points)
    joint_angles = trj.spline_trajectory(t_final, joint_targets, t_control=t)
    joint_velocities = trj.spline_trajectory(t_final, joint_targets, t_control=t, derivative=1)

    np.savez(
        Path(__file__).parents[1] / "trajectories" / out_file_name,
        time = t_final,
        joint_position = joint_angles,
        joint_velocity = joint_velocities,
        world_position = cartesian_targets,
    )


def plan_arch_camera(environment:Path, goal_idx:int, target_cam:str, out_file_name:str) -> None:
    num_plan_points = 300
    num_control_points = 50
    
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

    for value, joint in zip([0, -0.785, 0, -2.356, 0, 1.571, 0.785], joint_list):
        joint.param = value

    px_frame = {
        "main_camera": main_camera_frame,
        "angle_camera": angle_camera_frame,
        "side_camera": side_camera_frame,
    }[target_cam]

    camera_frame = {
        "main_camera": frames["main_camera::link::camera::camera"],
        "angle_camera": frames["angle_camera::link::camera::camera"],
        "side_camera": frames["side_camera::link::camera::camera"],
    }[target_cam]
    depth_frame = tf.CustomLink(3, 1, lambda x: x[..., 0])(camera_frame)


    start = tool_frame.transform((0,0,0), px_frame)
    end = goal_frames[goal_idx].transform((0,0, 0.03), px_frame)
    camera_targets = np.linspace(start, end, num_plan_points)

    start = tool_frame.transform((0,0,0), depth_frame)
    end = goal_frames[goal_idx].transform((0,0, 0.03), depth_frame)
    depth_targets = np.linspace(start, end, num_plan_points)

    goal_positions = np.stack([x.transform((0,0,0), px_frame) for x in goal_frames])
    displacement = 0*np.sin(np.linspace(0, np.pi, num_plan_points))

    gradients = np.sum(camera_targets[None, ...] - goal_positions[:, None, :], axis=0)
    gradients /= np.linalg.norm(gradients, axis=-1)[..., None]
    gradients *= displacement[..., None]
    gradients[[0,-1],:] = 0
    camera_targets += gradients

    joint_targets = np.zeros((num_plan_points, 9), dtype=float)
    for idx in range(num_plan_points):
        cam_pos = camera_targets[idx]
        depth_pos = depth_targets[idx]
        
        targets = [
            ik.PositionTarget((0,0,0), pos, tool_frame, px_frame),
            ik.PositionTarget((0,0,0), depth_pos, tool_frame, depth_frame),
        ]
        joint_targets[idx, :7] = ik.ccd(targets, [x for x in reversed(joint_list)])[::-1]

        for joint, value in zip(joint_list, joint_targets[idx, :7]):
            joint.angle = value

    t_final = np.linspace(0, 10, num_control_points)
    joint_angles = trj.spline_trajectory(t_final, joint_targets, t_control=t)
    joint_velocities = trj.spline_trajectory(t_final, joint_targets, t_control=t, derivative=1)

    np.savez(
        Path(__file__).parents[1] / "trajectories" / out_file_name,
        time = t_final,
        joint_position = joint_angles,
        joint_velocity = joint_velocities,
        world_position = camera_targets,
    )

if __name__ == "__main__":
    import imageio as iio
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    environment = Path(__file__).parent / "sdf" / "four_goals.sdf"
    # plan_direct(environment, 0, "straight.npz")
    # plan_arch(environment, 0, "arch.npz")
    plan_arch_camera(environment, 0, "main_camera", "arch_main_camera.npz")
