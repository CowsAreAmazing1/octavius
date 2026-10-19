"""testing a specific type of cube"""

from collections import namedtuple

import numpy as np

from ..cubes.datacube import DataCube, Axis

from ..data_management.data_structures import SimulationData
from ..data_management.conventions import OctaviusConfig

DensityCubeData = namedtuple(
    "DensityCubeData",
    [
        "pos",
        "mass",
        # "centre",
    ],
)

def make_density_cube(
    density_cube_data: DensityCubeData,
    side_length: float,
    resolution: int,
) -> DataCube:
    
    positions = np.asarray(density_cube_data.pos, dtype=np.float64)
    masses = np.asarray(density_cube_data.mass, dtype=np.float64)
    # centre = np.asarray(density_cube_data.centre, dtype=np.float64)
    
    edges = np.linspace(-side_length / 2, side_length / 2, resolution + 1)
    
    centre = np.mean(positions, axis=0)    
    rel_positions = positions - centre
    
    mass_grid, _ = np.histogramdd(
        rel_positions,
        bins=(edges, edges, edges),
        weights=masses,
    )

    voxel_width = side_length / resolution
    voxel_volume = voxel_width ** 3
    density_grid = mass_grid / voxel_volume
    
    axis_coord_centres = (edges[:-1] + edges[1:]) / 2
    
    return DataCube(
        data=density_grid,
        axes=(
            Axis(name="x", unit="kpc", coordinates=axis_coord_centres),
            Axis(name="y", unit="kpc", coordinates=axis_coord_centres),
            Axis(name="z", unit="kpc", coordinates=axis_coord_centres),
        ),
        quantity="density",
        unit="Msun/kpc^3",
    )
