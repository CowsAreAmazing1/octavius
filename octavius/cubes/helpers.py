
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R

def align_vectors_to_z(vectors: np.ndarray, alignment_vector: np.ndarray) -> np.ndarray:
    """
    Rotates a set of 3D vectors in a way which aligns the given `alignment_vector` with the z-axis.

    Parameters
    ----------
    vectors : np.ndarray
        The 3D vectors to be aligned. Should have shape (N, 3).
    alignment_vector : np.ndarray
        The vector to be aligned with the z-axis. The `vectors` will be rotated the same way.

    Returns
    -------
    np.ndarray
        The aligned vectors.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    alignment_vector = np.asarray(alignment_vector, dtype=np.float64)
    
    if vectors.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of length 3, representing 3D vectors.")
    if alignment_vector.shape != (3,):
        raise ValueError("alignment_vector must have shape (3,), a single 3D vector")

    alignment_norm = np.linalg.norm(alignment_vector)
    if alignment_norm == 0 or not np.isfinite(alignment_norm):
        raise ValueError("alignment_vector must be finite and non-zero.")
    alignment_unit = alignment_vector / alignment_norm
    z_axis = np.array([0.0, 0.0, 1.0])

    cos_angle = np.clip(np.dot(alignment_unit, z_axis), -1.0, 1.0)
    rotation_axis = np.cross(alignment_unit, z_axis)
    axis_norm = np.linalg.norm(rotation_axis)

    if axis_norm < 1e-12:
        if cos_angle > 0:
            return vectors.copy()
        rotation = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        rotation_axis /= axis_norm
        rotation_angle = np.arccos(cos_angle)
        rotation = R.from_rotvec(rotation_angle * rotation_axis)

    return rotation.apply(vectors)

# @njit(cache=True)
def build_particle_cell_list(
    pos: np.ndarray,
    smoothing_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int, float, float, float, float]:
    """
    I think this is for efficiently indexing particles in an existing list, in a way that allows
    consideration of each particle's neighbors, up to a maximum distance equal to the largest? smoothing length
    
    Creates a cell linked list for dust attenutation. Returns:

    - sort_order: the indices to sort particles by cell (np.argsort)
    - cell_offsets: where each cell begins in the flat sorted array (classic csr offset)
    - n_cells_{x/y}: the number of cells in the orthogonal directions
    - origin_{x/y}: the origin of the cell list in the orthogonal directions
    - inv_cell_width: inverse of the width of each cell (sized to max smoothing length)
    """
    if len(pos) == 0:  # you can't hit this in principle, but I'm leaving it here in case the code changes in future
        empty = np.empty(0, dtype=np.int64)
        offsets = np.zeros(1, dtype=np.int64)
        return empty, offsets, 0, 0, 0, 0.0, 0.0, 0.0, 0.0  # match type check

    # set the maximum cell width to h_max so a star will always access all gas which contributes to its attenutation
    inv_cell_width = 1 / np.max(
        smoothing_length
    )  # optimisation: compute inverse here then use faster multiplication instruction

    # np.min/max with axis arg is not supported in numba (yet)
    origin_x = np.min(pos[:, 0])
    origin_y = np.min(pos[:, 1])
    origin_z = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    max_z = np.max(pos[:, 2])
    n_cells_x = int((max_x - origin_x) * inv_cell_width) + 1  # padding so particles at the edge are always included
    n_cells_y = int((max_y - origin_y) * inv_cell_width) + 1
    n_cells_z = int((max_z - origin_z) * inv_cell_width) + 1

    n_particles = len(pos)
    cell_ids = np.empty(
        shape=n_particles, dtype=np.int32
    )  # NOTE: int32 should suffice here since this is performance-critical

    for i in range(n_particles):
        ix = int((pos[i, 0] - origin_x) * inv_cell_width)
        iy = int((pos[i, 1] - origin_y) * inv_cell_width)
        iz = int((pos[i, 2] - origin_z) * inv_cell_width)
        cell_idx = ix * n_cells_y * n_cells_z + iy * n_cells_z + iz  # row-major ordering; z,y,x in order of increasing significance
        cell_ids[i] = cell_idx

    sort_order = np.argsort(cell_ids, kind="quicksort")  # NOTE: dense, not sparse, due to the geometry of the cells
    n_total_cells = n_cells_x * n_cells_y * n_cells_z
    cell_offsets = np.zeros(n_total_cells + 1, dtype=np.int64)

    # add number of particles in each cell (offsets becomes the number of particles in cell i-1)
    for i in range(n_particles):
        cell_offsets[cell_ids[i] + 1] += 1

    # then add per-cell counts to the next offset so offsets becomes actually particle-indexable
    for i in range(n_total_cells):
        cell_offsets[i + 1] += cell_offsets[i]

    return sort_order, cell_offsets, n_cells_x, n_cells_y, n_cells_z, origin_x, origin_y, origin_z, inv_cell_width
