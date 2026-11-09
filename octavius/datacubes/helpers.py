"""
Generic functions used throughout datacube generation and manipulation.
"""

# other packages
import numpy as np
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
