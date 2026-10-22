
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
    alignment_vector = alignment_vector / np.linalg.norm(alignment_vector)
    
    z_axis = np.array([0, 0, 1])
    
    rotation_axis = np.cross(z_axis, alignment_vector)
    rotation_angle = np.arccos(np.clip(np.dot(z_axis, alignment_vector), -1.0, 1.0))
    
    if np.linalg.norm(rotation_axis) < 1e-8:
        return vectors
    
    rotation_axis /= np.linalg.norm(rotation_axis)

    # ?    
    rotation = R.from_rotvec(rotation_angle * rotation_axis)
    
    aligned_vectors = rotation.apply(vectors)
    
    return aligned_vectors
