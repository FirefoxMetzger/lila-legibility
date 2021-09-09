import numpy as np
from numpy.typing import ArrayLike
from typing import List


def cost(trajectory: ArrayLike, goals: List[ArrayLike]) -> List[float]:
    """Path Length Cost - lower is better

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
