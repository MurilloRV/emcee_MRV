# -*- coding: utf-8 -*-

import numpy as np
inner = np.inner
norm = np.linalg.norm

from .red_blue import RedBlueMove
from ..utils import rng_integers

__all__ = ["ModifiedStretchMove"]


class ModifiedStretchMove(RedBlueMove):
    """
    A modified version of the `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "stretch move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_.

    At each iteration, instead of choosing an arbitrary walker from the 
    complementary ensemble in order to define the stretch direction, this 
    choice is made arbitrarily but such that the line spanned by the current and 
    the complementary walkers must be at least at a certain distance away from the
    likelihood maximum. This distance is calculated by first performing a change
    of variables, using an estimate of the covariance matrix of the likelihood, 
    under which a Gaussian likelihood would be changed into a Gaussian with covariance
    equal to the identity matrix. If the complementary walker is sufficiently close
    to the current one (i.e. inside a tolerance region), this walker will be used for 
    the proposal of the next iteration in any case.

    Args:
        x_max (ndarray[ndim])
            Location of the likelihood maximum
        cov_inv (ndarray[ndim, ndim])
            Estimate of the inverse of the sample covariance matrix
        d1 (Optional[float])
            Minimum distance between the line spanned by complementary walker (used 
            to propose the next iteration step) and the current walker, and the 
            location of the likelihood maximum
        d2 (Optional[float])
            Radius of the tolerance region around the current walker. If the
            complementary walker has a (normalized) distance to the current
            walker less than "d2", such complementary walker will be used to propose
            the next iteration step in any case, regardless of "d1"
        a (optional)
            The stretch scale parameter. (default: ``2.0``)

    """

    def __init__(self, x_max, cov_inv, a=2.0, d1=0, d2=0, **kwargs):
        self.a = a
        self.x_max = x_max
        self.cov_inv = cov_inv
        self.d1 = d1
        self.d2 = d2
        super(ModifiedStretchMove, self).__init__(**kwargs)

    def check_distance(self, x1, x2):
        # Distance vector between the line spanned by x1-x2 and the likelihood maximum (self.x_max)
        d = -(self.x_max - x1) + (inner(self.x_max-x1, x2-x1) / (norm(x2-x1))**2) * (x2-x1)
        line_distance = d.T @ self.cov_inv @ d

        # Normalized distance between the two walkers
        x1_x2_distance = (x2-x1).T @ self.cov_inv @ (x2-x1)

        if line_distance <= self.d1 and x1_x2_distance > self.d2 : 
            return False
        else:
            return True

    def get_proposal(self, s, c, random):
        c = np.concatenate(c, axis=0)
        Ns, Nc = len(s), len(c)
        ndim = s.shape[1]
        if np.array([ndim]) != self.x_max.shape[0]:
            raise ValueError(f"Mismatch between the dimensions of the likelihood maximum {self.x_max.shape}"
                             f"and the current sample {ndim}")
        
        zz = random.uniform(low=1, high=self.a, size=Ns) ** 2.0 / self.a
        factors = (ndim - 1.0) * np.log(zz)

        rint = np.empty(Ns, dtype=int)
        for w1 in range(Ns):
            for i, w2 in enumerate(random.permutation(Nc)):
                # Loops over the walkers in the complementary ensemble, until one far enough
                # from the LL maximum is found or until the last available walker is considered
                if self.check_distance(s[w1], c[w2]) or i==Nc-1:
                    rint[w1] = w2
                    break

        return c[rint] - (c[rint] - s) * zz[:, None], factors