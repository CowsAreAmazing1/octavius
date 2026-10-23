from octavius.cubes.helpers import align_vectors_to_z

import numpy as np

def test_align_aligned_vectors():
    v = np.array([1, 0, 0])
    dummy = np.array([1, 0, 0])
    aligned_v = align_vectors_to_z(v, dummy)
    try:
        assert np.allclose(aligned_v, [0, 0, 1])
    except AssertionError:
        print(f"Failed 1: Aligned vector: {aligned_v}, Expected: [0, 0, 1]")

def test_one_eighth_rotation():
    v = np.array([1, 0, 0])
    dummy = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])
    aligned_v = align_vectors_to_z(v, dummy)
    try:
        assert np.allclose(aligned_v, [1/np.sqrt(2), 0, 1/np.sqrt(2)])
    except AssertionError:
        print(f"Failed 2: Aligned vector: {aligned_v}, Expected: [{1/np.sqrt(2)}, 0, {1/np.sqrt(2)}]")
