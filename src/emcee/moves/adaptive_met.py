# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

import warnings
#warnings.filterwarnings("error")

__all__ = ["AdaptiveMetropolisMove"]


class AdaptiveMetropolisMove(Move):
    r"""
    Implementation of the Adaptive Metropolis algorithm proposed by Haario, 
    Saksman and Tamminen: Bernoulli, Vol. 7, No. 2 (Apr., 2001), pp. 223-242 
    <https://doi.org/10.2307/3318737>

    Args:
        ndim: Number of dimensions of the parameter space
        C0: Initial covariance matrix
        Sd (Optional): Multiplicative factor in the definition of the 
            covariance matrix at a later iteration (Harrio et al.). If not
            given, the default value of :math:`(2.4)^2/ndim` will be used.
        t0 (Optional[Int]): Number of initial iterations where the covariance
            matrix remains unchained.
        epsilon (Optional): Small factor multiplying the identity matrix in 
            the definition of the covariance matrix at a later iteration 
            (Harrio et al.). Must be much smaller than the dimensions of the 
            parameter space
        update_all (Optional[bool]): If `True`, the covariance matrix will be
            updated using all points in the sample. If `False`, the matrix is
            only updated using AM move points.


    """

    def __init__(
            self, ndim, C0, epsilon, Sd=None, t0=1, update_all=False
        ):

        if Sd is None: Sd = ((2.4)**2)/float(ndim)
        if not C0.shape == (ndim, ndim):
            raise ValueError("Mismatch between the shape of C0 = {0} and ndim={1}".format(C0.shape, ndim))
        if t0 < 1:
            raise ValueError("t0 must be a positive integer!") 

        self.ndim = ndim
        self.C0 = C0
        self.Sd = Sd
        self.t0 = t0
        self.epsilon = epsilon
        self.update_all = update_all

    def get_proposal(self, x0, C_t, random):
        
        nwalkers, ndim = x0.shape
        return np.array([x0[i] + random.multivariate_normal(np.zeros(len(C_t[i])), C_t[i]) 
                            for i in range(nwalkers)]
                            )

    def propose(self, model, state):

        # Check to make sure that the dimensions match.
        nwalkers, ndim = state.coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError("Dimension mismatch in proposal")
        
        # Assign the initial values of C_t and Xbar_t to the initial state
        if state.C_t is None: state.C_t = np.full((nwalkers, ndim, ndim), self.C0)
        if state.Xbar_t is None: state.Xbar_t = state.coords

        # Get the actual AM proposal
        q = self.get_proposal(state.coords, state.C_t, model.random)

        # Compute the lnprobs of the proposed position.
        new_log_probs, new_blobs = model.compute_log_prob_fn(q)

        # Loop over the walkers and update them accordingly.
        lnpdiff = new_log_probs - state.log_prob
        accepted = np.log(model.random.rand(nwalkers)) < lnpdiff

        # Update the parameters
        new_state = State(q, log_prob=new_log_probs, blobs=new_blobs, C_t=state.C_t, Xbar_t=state.Xbar_t)
        state = self.update(state, new_state, accepted)

        return state, accepted, new_state

