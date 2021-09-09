""" Implementation of Dragan's Legibility

Based on the Paper: Generating Legible Motion
https://kilthub.cmu.edu/articles/journal_contribution/Generating_Legible_Motion/6554969/1

This version does not implement the mentioned trust-region optimization, i.e.,
the optimization method is not included. Instead, it only implements the
objective function, which can then be used with your favourite optimization
differentiation tool (torch, tensorflow, jax, scipy, ...).

This version also generalizes the final step of the objective function. Instead
of providing a single score (range [0, 1]) for the trajectory and the current
goal, the entire distribution over goals is returned as well as a legibility
estimate for each point along the trajectory. To recover the single value used
in the paper, one can use ``legibility(...)[-1, idx]`` where idx referrs to the
true goal of the motion.

"""

import numpy as np
from numpy.typing import ArrayLike
import skbot.trajectory as rtj


def velocity_fast(
    t: ArrayLike,
    keyframes: ArrayLike,
    *,
    trajectory_axis: float = -2,
    space_axis: int = -1
) -> np.ndarray:
    """Compute the velocity along a path.

    This function uses finite differences to approximate the velocity
    along a time-parameterized curve. Every dimension that isn't the
    trajectory_axis or space_axis is considered a batch dimension.

    Parameters
    ----------
    t : ArrayLike
        The time points at which the path was evaluated.
    keyframes : ArrayLike
        The corrdinates of the path at the times it was evaluated.
    trajectory_axis : float
        The axis along which to evaluate the velocity.
    space_axis : float
        The axis along which the coordinates of the path are stored.

    Returns
    -------
    velocity : np.ndarray
        The velocities of the given trajectories.

    """

    t = np.expand_dims(t, -1)
    t = np.moveaxis(t, trajectory_axis, -2)
    keyframes = np.moveaxis(keyframes, (trajectory_axis, space_axis), (-2, -1))

    delta_t = np.diff(t, axis=-2, append=np.inf)
    direction = np.diff(keyframes, axis=-2, append=0)
    velocity = np.nan_to_num(direction / delta_t)

    velocity = np.moveaxis(velocity, (-2, -1), (trajectory_axis, space_axis))

    return velocity


def cost(
    keyframes: ArrayLike, t: ArrayLike, *, integration_axis=-2, space_axis=-1
) -> np.ndarray:
    """The human's attributed cost of a trajectory.

    Uses the cost model proposed by Dragan et al. to compute the cost of a given
    set of trajectories. The cost is the integral of the square norm of the
    trajectories velocity. Integration axis controlls the dimension along which
    the integral is computed and space axis controls the dimension along with
    the norm is computed.

    Parameters
    ----------
    keyframes : ArrayLike
        The control points describing the trajectories. Typically of the form
        [..., trajectory, space]
    t : ArrayLike
        The time points at which each keyframe is reached. Must be broadcastable
        to keyframes.shape.
    integration_axis : int
        The axis along which the trajectory progresses.
    space_axis : int
        The axis along which the space values are stored.

    Returns
    -------
    scores : np.ndarray
        The cost of the trajectory/trajectories described by the keyframes.

    """

    keyframes = np.moveaxis(keyframes, (integration_axis, space_axis), (-2, -1))

    velocity = velocity_fast(t, keyframes)
    square_norm_v = np.sum(velocity ** 2, axis=-1)
    return rtj.utils.integral(square_norm_v, t, axis=-1)


def V_fast(
    start: ArrayLike, goal: ArrayLike, velocity: float, *, space_axis: float = -1
) -> np.ndarray:
    """Compute the cost of the minimum cost trajectory to the goal.

    This is a computationally efficient version that makes _strong_ assumptions about the cost function.
    I.e., it assumes the cost function to be exactly the one specified in the paper by Dragan et al.
    It also assume that the robot travels with constant velocity

    Parameters
    ----------
    start : ArrayLike
        The starting positions of the trajectory. start[space_axis] contains the
        coordinates, and all other dimensions are considered batch dimensions.
    goal : ArrayLike
        The goal positions where the trajectory may lead. goal[space_axis] contains
        the coordinates, and all other dimensions are considered batch dimensions.
    velocity : float
        The velocity with which the robot travels (assumed to be constant).
    space_axis : int
        The axis along with the coordinate values are stored.

    Returns
    -------
    expected_costs : np.ndarray
        The cost of moving from start to goal. This array has shape (*start_batch.shape, *goal_batch.shape)

    """

    start = np.moveaxis(start, space_axis, -1)
    goal = np.moveaxis(goal, space_axis, 0)
    space_axis = start.ndim - 1

    start = np.expand_dims(start, (start.ndim + np.arange(goal.ndim - 1)).tolist())
    goal = np.expand_dims(goal, np.arange(space_axis).tolist())

    time = np.linalg.norm(goal - start, axis=space_axis) / velocity
    time = np.linspace(np.zeros_like(time), time, 2, axis=-1)
    values = np.linspace(start, goal, 2, axis=-1)
    values = np.moveaxis(values, space_axis, -1)

    return cost(values, time)


def prop_goal(
    t: ArrayLike,
    control_points: ArrayLike,
    goals: ArrayLike,
    velocity: float,
    *,
    weights: ArrayLike = None
) -> np.ndarray:
    """Calculate the probability of moving to each goal from the current position.

    Parameters
    ----------
    t : ArrayLike
        The time points at which to estimate the distribution.
    control_points : ArrayLike
        The coordinates of the trajectory at times t.
    goals : ArrayLike
        A sequence of potential goals to move towards.
    velocity : ArrayLike
        The magnitude of the velicity to use when planning future trajectories.
    weights : ArrayLike
        The prior probability of moving towards each goal in goals. If None
        it will be set to ``np.ones(len(goals)) / len(goals)``

    Returns
    -------
    p_dist : np.ndarray
        A array of probabilities, where p_dist[x] corresponds to goals[x].
    """

    start = control_points[0]

    if weights is None:
        weights = np.full(len(goals), 1 / len(goals))

    expected_cost = V_fast(control_points, goals, velocity)
    cost_from_start = V_fast(start[None, ...], goals, velocity)

    vel = velocity_fast(t, control_points)
    square_norm_v = np.sum(vel ** 2, axis=-1)
    cost_so_far = rtj.utils.cumulative_integral(square_norm_v, t)[..., None]

    numerator = np.exp(cost_from_start - expected_cost - cost_so_far)
    p_dist = numerator * weights[None, ...]
    p_dist /= np.sum(p_dist, axis=-1, keepdims=True)
    return p_dist


def legibility(
    control_points: ArrayLike,
    potential_goals: ArrayLike,
    velocity: float,
    *,
    t_control: ArrayLike = None
) -> float:
    """Compute the Dragan-Legibility along a given trajectory.

    The trajectory is given by a sequence of control points (keyframes). Control
    point positioning along the trajectory can be controlled by passing
    t_control.

    Parameters
    ----------
    control_points : ArrayLike
        A sequence of points (keyframes) describing the trajectory.
        ``control_points[0]`` denotes the starting point of the trajectory, and
        ``control_points[-1]`` denotes the end point of the trajectory.
    potential_goals : ArrayLike
        A sequence of potential goals the trajectory may lead to.
    velocity : ArrayLike
        The velocity used to estimate V(position, goal) (the lowest cost trajectory
        from a given point to a goal).
    t_control : ArrayLike
        A sequence of time points at which a corresponding control point is
        reached. If None, this will be set to ``np.linspace(0, 1,
        len(control_points))``.

    Returns
    -------
    score : ndarray
        The Dragan-Legibility of the trajectory. The probability of the motion
        ending in each potential goal is returned for each control point along
        the trajectory (except the first - there is no trajectory).

    """

    if t_control is None:
        t_control = np.linspace(0, 1, len(control_points))

    t_start = t_control[0]
    t_end = t_control[-1]

    t = np.linspace(t_start, t_end, 100)
    eval_points = rtj.linear_trajectory(t, control_points, t_control=t_control)
    goal_odds = prop_goal(t, eval_points, potential_goals, velocity)
    scale_factor = t_end - t

    numerator = rtj.utils.cumulative_integral(
        goal_odds * scale_factor[:, None], t[:, None]
    )[1:]
    denominator = rtj.utils.cumulative_integral(scale_factor, t)[1:]

    legibility_scores = numerator / denominator[:, None]

    return legibility_scores
