from copy import deepcopy
from pathlib import Path
import random

import click
import numpy as np
from skbot.ignition import sdformat
from skbot.ignition.sdformat.bindings import v18


def sample_positions(
    num_samples: int,
    *,
    max_samples: int = 1000,
    min_distance: float = 0.1,
    seed: int = None,
):
    """Rejection sample a set of cube positions

    The positions are all at least ``min_distance`` apart. With the default
    setting this means 5x5x5 cm cubes can't touch each other.
    """

    rng = np.random.RandomState(seed)

    if num_samples == 0:
        return np.array([])

    center = np.array((0.4, 0, 0.025))
    half_extent = np.array((0.25, 0.5, 0))

    accepted_samples = np.empty((num_samples, 3))
    accepted_samples[0] = center + rng.rand(3) * 2 * half_extent - half_extent
    num_accepted = 1

    samples_per_step = 3
    samples_generated = 0
    while num_accepted < num_samples and samples_generated < max_samples:
        samples = (
            center[:, None]
            + rng.rand(3, samples_per_step) * 2 * half_extent[:, None]
            - half_extent[:, None]
        )
        samples_generated += samples_per_step

        while samples.shape[1] > 0:
            actual_samples = accepted_samples[:num_accepted]
            # remove samples closer than 0.1 to any accepted pos
            distance = np.linalg.norm(
                samples[None, ...] - actual_samples[..., None], axis=1
            )
            distance_ok = np.all(distance >= min_distance, axis=0)
            samples = samples[:, distance_ok]

            # add the first stample if any are left
            if samples.shape[1] > 0:
                accepted_samples[num_accepted] = samples[:, 0]
                num_accepted += 1

                if num_accepted >= num_samples:
                    break

    return accepted_samples


@click.command()
@click.option(
    "--num-goals",
    "num_goals",
    default=5,
    help="The number of goals/cubes to be placed in the environment.",
    type=click.IntRange(4, 6),
)
@click.option("--seed", default=None, help="The seed to use for rng.", type=click.INT)
def generate_environment(num_goals: int, seed: int) -> None:
    sdf_folder = Path(__file__).parent / "sdf"
    env: v18.Sdf = sdformat.loads((sdf_folder / "environment_template.sdf").read_text())
    cube_base: v18.ModelModel = sdformat.loads(
        (sdf_folder / "cube_template.sdf").read_text()
    ).model

    rng = random.Random(seed)

    world = env.world[0]

    positions = sample_positions(num_goals, seed=seed)

    colors = [
        # faithfully borrowed from https://xkcd.com/color/rgb/
        (126, 30, 156, 1),  # purple
        (21, 176, 26, 1),   # green
        (3, 67, 223, 1),    # blue
        (255, 129, 192, 1), # pink
        (149, 208, 252, 1), # teal
        (249, 115, 6, 1),   # orange
    ]

    for cube_idx in range(num_goals):
        cube: v18.ModelModel = deepcopy(cube_base)
        cube.name = cube.name + f"_copy_{cube_idx}"
        world.model.append(cube)
        cube.pose.relative_to = "tabletop_2"
        cube_position = " ".join([str(x) for x in positions[cube_idx]])
        cube.pose.value = cube_position + " 0 0 0"

        color = colors.pop(rng.randint(0, len(colors) - 1))
        cube.link[0].visual[0].material.ambient = color
        cube.link[0].visual[0].material.diffuse = color
        cube.link[0].visual[0].material.specular = color

    sdf_string = sdformat.dumps(env, format=True)

    print(f"<!-- Generated using Lila Legibility v1.0. Seed: {seed} -->")
    print(sdf_string)
