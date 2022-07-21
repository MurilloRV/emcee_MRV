# -*- coding: utf-8 -*-

import numpy as np

from ..state import State
from .move import Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move):
    """
    An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. Each
            sub-ensemble is updated in parallel using the other sets as the
            complementary ensemble. The default value is ``2`` and you
            probably won't need to change that.

        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles. The same number of walkers will be assigned to
            each sub-ensemble on each iteration. By default, this is ``True``.

        live_dangerously (Optional[bool]): By default, an update will fail with
            a ``RuntimeError`` if the number of walkers is smaller than twice
            the dimension of the problem because the walkers would then be
            stuck on a low dimensional subspace. This can be avoided by
            switching between the stretch move and, for example, a
            Metropolis-Hastings step. If you want to do this and suppress the
            error, set ``live_dangerously = True``. Thanks goes (once again)
            to @dstndstn for this wonderful terminology.

    """

    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False
    ):
        self.nsplits = int(nsplits)
        self.live_dangerously = live_dangerously
        self.randomize_split = randomize_split

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError(
            "The proposal must be implemented by " "subclasses"
        )

    def propose(self, model, state):
        """Use the move to generate a proposal and compute the acceptance

        Args:
            coords: The initial coordinates of the walkers.
            log_probs: The initial log probabilities of the walkers.
            log_prob_fn: A function that computes the log probabilities for a
                subset of walkers.
            random: A numpy-compatible random number state.

        """
        # Check that the dimensions are compatible.
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of "
                "dimensions."
            )

        # Run any move-specific setup.
        self.setup(state.coords)

        # Split the ensemble in half and iterate over these two halves.
        accepted = np.zeros(nwalkers, dtype=bool)
        accepted_full = np.ones(nwalkers, dtype=bool) #boolean vector, where all walkers are accepted
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
            
        # Initializing the variables which will contain the full state information, without discarding proposals
        #q_full = np.empty([nwalkers, ndim]) # Too complex
        #new_log_probs_full = np.empty([nwalkers]) # Too complex
        #new_blobs_full = state.blobs # Too complex
        
        for split in range(self.nsplits):
            S1 = inds == split

            # Get the two halves of the ensemble.
            sets = [state.coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = sets[:split] + sets[split + 1 :]

            # Get the move-specific proposal.
            q, factors = self.get_proposal(s, c, model.random)
            #print(f'q = {q}, factors = {factors}') #flag
            #print(f'type of q = {type(q)}') #flag
            #print(f'shape of q = {q.shape}') #flag
            
            
            #q_full[S1, :] = q  # Too complex

            # Compute the lnprobs of the proposed position.
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)
            
            #print(f'type of new_log_probs = {type(new_log_probs)}') #flag
            #print(f'shape of new_log_probs = {new_log_probs.shape}') #flag
            #print(f'type of new_blobs = {type(new_blobs)}') #flag
            #print(f'shape of new_blobs = {new_blobs.shape}') #flag
            #print(f'blobs = {new_blobs}') #flag
            
            #new_log_probs_full[S1] = new_log_probs # Too complex
            #new_blobs_full[S1] = new_blobs # Too complex
            
            
            

            # Loop over the walkers and update them accordingly.
            for i, (j, f, nlp) in enumerate(
                zip(all_inds[S1], factors, new_log_probs)
            ):
                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True

            new_state_prelim = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state_prelim, accepted, S1)
            
            new_state = self.update(state, new_state_prelim, accepted_full, S1) # this state now contains all walker proposals
        
        #print(f'q_full = {q_full}') #flag
        #print(f'new_state_prime = {new_state_prime.coords}') #flag
        #print(f'new_log_probs_full = {new_log_probs_full}') #flag
        #print(f'new_state_prime_log = {new_state_prime.log_prob}') #flag
        #print(f'new_blobs_full = {new_blobs_full}') #flag
        #print(f'new_state_prime_blobs = {new_state_prime.blobs}') #flag
        
        #new_state_prime = State(q_full, log_prob=new_log_probs_full, blobs=new_blobs_full) # Too complex
        #print(f'Are the two different ways to get the full state equivalent: {new_state==new_state_prime}') #flag
        #print(f'Are the two q equivalent: {new_state.coords==new_state_prime.coords}') #flag
        #print(f'Are the two LLI equivalent: {new_state.log_prob==new_state_prime.log_prob}') #flag
        #print(f'Are the two blobs equivalent: {new_state.blobs==new_state_prime.blobs}') #flag
        #print(f'Are the two random states equivalent: {new_state.random_state==new_state_prime.random_state}') #flag
        
        return state, accepted, new_state
