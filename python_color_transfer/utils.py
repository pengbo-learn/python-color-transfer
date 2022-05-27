# -*- coding: utf-8 -*-

import numpy as np


def rvs(dim=3):
    """generate orthogonal matrices with dimension=dim.

    This is the rvs method pulled from the https://github.com/scipy/scipy/pull/5622/files,
    with minimal change - just enough to run as a stand alone numpy function.
    """
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim, ))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1, ))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


class Rotations:
    """ generate orthogonal matrices for pdf transfer."""

    @classmethod
    def random_rotations(cls, m, c=3):
        """ Random rotation. """

        assert m > 0
        rotation_matrices = [np.eye(c)]
        rotation_matrices.extend([
            np.matmul(rotation_matrices[0], rvs(dim=c)) for _ in range(m - 1)
        ])
        return rotation_matrices

    @classmethod
    def optimal_rotations(cls):
        """Optimal rotation.

        Copy from Automated colour grading using colour distribution transfer.
        F. Pitié , A. Kokaram and R. Dahyot (2007) Journal of Computer Vision and Image Understanding.
        """

        rotation_matrices = [
            [
                [1.000000, 0.000000, 0.000000],
                [0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 1.000000],
            ],
            [
                [0.333333, 0.666667, 0.666667],
                [0.666667, 0.333333, -0.666667],
                [-0.666667, 0.666667, -0.333333],
            ],
            [
                [0.577350, 0.211297, 0.788682],
                [-0.577350, 0.788668, 0.211352],
                [0.577350, 0.577370, -0.577330],
            ],
            [
                [0.577350, 0.408273, 0.707092],
                [-0.577350, -0.408224, 0.707121],
                [0.577350, -0.816497, 0.000029],
            ],
            [
                [0.332572, 0.910758, 0.244778],
                [-0.910887, 0.242977, 0.333536],
                [-0.244295, 0.333890, -0.910405],
            ],
            [
                [0.243799, 0.910726, 0.333376],
                [0.910699, -0.333174, 0.244177],
                [-0.333450, -0.244075, 0.910625],
            ],
            # [[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]],
            # [[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]],
            # [[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]],
            # [[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]],
            # [[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]],
            # [[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]],
        ]
        rotation_matrices = [np.array(x) for x in rotation_matrices]
        # for x in rotation_matrices:
        #    print(np.matmul(x.transpose(), x))
        #    import pdb
        #    pdb.set_trace()
        return rotation_matrices
