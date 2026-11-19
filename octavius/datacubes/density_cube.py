"""
Functions for generating and manipulating 3D spatial density datacubes.

Deals with 3D grid sampling and other data handling.
"""

# default packages
from dataclasses import dataclass, field

# other packages
import numpy as np
from numba import njit

# internal imports
from .datacube import Axis, DataCube
from .helpers import align_vectors_to_z
from ..data_management.conventions import OctaviusConfig
from ..photometry.photometry_helpers import KERNEL_INT_MAP, _cubic_kernel, _quintic_kernel

@njit(cache=True)
def _deposit_property_by_density(
    positions: np.ndarray,
    property: np.ndarray,
    smoothing_lengths: np.ndarray,
    lower_bounds: np.ndarray,
    widths: np.ndarray,
    resolution: int,
    kernel_type: int,
) -> np.ndarray:
    """Sample smoothed particle density onto a 3D grid."""
    density_grid = np.zeros((resolution, resolution, resolution), dtype=np.float64)

    for particle_index in range(len(positions)):
        position = positions[particle_index]
        smoothing_length = smoothing_lengths[particle_index]
        lower_x = max(0, int(np.floor((position[0] - smoothing_length - lower_bounds[0]) / widths[0])))
        lower_y = max(0, int(np.floor((position[1] - smoothing_length - lower_bounds[1]) / widths[1])))
        lower_z = max(0, int(np.floor((position[2] - smoothing_length - lower_bounds[2]) / widths[2])))
        upper_x = min(resolution, int(np.ceil((position[0] + smoothing_length - lower_bounds[0]) / widths[0])))
        upper_y = min(resolution, int(np.ceil((position[1] + smoothing_length - lower_bounds[1]) / widths[1])))
        upper_z = min(resolution, int(np.ceil((position[2] + smoothing_length - lower_bounds[2]) / widths[2])))

        for ix in range(lower_x, upper_x):
            x = lower_bounds[0] + (ix + 0.5) * widths[0]
            dx = x - position[0]
            for iy in range(lower_y, upper_y):
                y = lower_bounds[1] + (iy + 0.5) * widths[1]
                dy = y - position[1]
                for iz in range(lower_z, upper_z):
                    z = lower_bounds[2] + (iz + 0.5) * widths[2]
                    q = np.sqrt(dx * dx + dy * dy + (z - position[2]) ** 2) / smoothing_length
                    if kernel_type == 0:
                        weight = _cubic_kernel(q)
                    else:
                        weight = _quintic_kernel(q)
                    density_grid[ix, iy, iz] += (
                        property[particle_index] * weight / smoothing_length**3
                    )
                    
    return density_grid


@dataclass(frozen=True)
class DensityCubeData:
    """A simple data structure to hold and handle the data needed to make a spatial density cube."""
    
    pos: np.ndarray # positions of the particles
    property: np.ndarray # desired property of the particles, 
    smoothing_length: np.ndarray | None
    com: np.ndarray # center of mass of the particles
    config: OctaviusConfig
    angular_momentum: np.ndarray | None = field(default=None) # angular momentum of the particles, optional
    boxsize: float | None = field(default=None) # box size of the simulation, optional

    def make_smoothed_density_cube(
        self,
        side_length: float | None = None,
        resolution: int = 256,
        padding: float = 0.0,
    ) -> DataCube:
        """
        Deposit particle masses, smoothed by the configured kernel.

        If `side_length` is omitted, each axis is fitted independently to the
        particle positions plus their kernel support and `padding`. This
        avoids empty space for flattened systems, at the cost of allowing
        rectangular voxels. If it is supplied, the cube is centred on `com`.
        """
        if len(self.pos) == 0:
            raise ValueError("No particles to make a density cube from.")
        if resolution < 1 or padding < 0:
            raise ValueError("resolution must be at least 1 and padding cannot be negative.")
        if side_length is not None and side_length <= 0:
            raise ValueError("side_length must be positive when supplied.")
        if self.smoothing_length is None:
            raise ValueError("smoothing_length is required for smoothed density deposition.")
        
        positions = np.asarray(self.pos, dtype=np.float64)
        smoothing_lengths = np.asarray(self.smoothing_length, dtype=np.float64)

        positions = positions - np.asarray(self.com, dtype=np.float64)
        
        if positions.shape != (len(self.property), 3):
            raise ValueError("pos must have shape (N, 3) and match mass.")
        if smoothing_lengths.shape != (len(self.property),):
            raise ValueError("smoothing_length must have shape (N,) and match mass.")
        if np.any(~np.isfinite(smoothing_lengths)) or np.any(smoothing_lengths <= 0):
            raise ValueError("smoothing_length must contain finite, positive values.")

        if self.angular_momentum is not None:
            positions = align_vectors_to_z(positions, self.angular_momentum)            

        if side_length is None:
            lower_bounds = np.min(positions - smoothing_lengths[:, None], axis=0) - padding
            upper_bounds = np.max(positions + smoothing_lengths[:, None], axis=0) + padding
        else:
            lower_bounds = np.full(3, -side_length / 2, dtype=np.float64)
            upper_bounds = np.full(3, side_length / 2, dtype=np.float64)

        widths = (upper_bounds - lower_bounds) / resolution
        if np.any(widths <= 0) or np.any(~np.isfinite(widths)):
            raise ValueError("The requested cube has invalid bounds.")
        outside = np.any(
            (positions + smoothing_lengths[:, None] < lower_bounds)
            | (positions - smoothing_lengths[:, None] > upper_bounds),
            axis=1,
        )
        if np.any(outside):
            raise ValueError(
                f"{np.count_nonzero(outside)} particle(s) lie entirely outside the requested cube. "
                "Omit side_length to fit the cube automatically, or increase side_length."
            )
        
        kernel_type = KERNEL_INT_MAP[self.config.kernel_type]
        
        density_grid = _deposit_property_by_density(
            positions=positions,
            property=self.property,
            smoothing_lengths=smoothing_lengths,
            lower_bounds=lower_bounds,
            widths=widths,
            resolution=resolution,
            kernel_type=kernel_type,
        )
        
        edges = tuple(
            np.linspace(lower_bounds[axis], upper_bounds[axis], resolution + 1)
            for axis in range(3)
        )
        centres = tuple((axis_edges[:-1] + axis_edges[1:]) / 2 for axis_edges in edges)
        

        return DataCube(
            data=density_grid,
            axes=(
                Axis(name="x", unit="kpc", coordinates=centres[0]),
                Axis(name="y", unit="kpc", coordinates=centres[1]),
                Axis(name="z", unit="kpc", coordinates=centres[2]),
            ),
            quantity="density",
            unit="Msun/kpc^3",
        )
