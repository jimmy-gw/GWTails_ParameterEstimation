'''Jump proposals for MCMC.'''


import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import wave_gen as wg
import likelihood as l



# Fisher jumps
class Fisher:

    def __init__(self, x0, _Lambda):
        self.x0 = x0  # initial state where to compute Fisher
        self.get_Fisher_func = l.get_Fisher  # function to get Fisher (fast: Cornish 2013 https://arxiv.org/pdf/1007.4820)
        self.history = None # no adaptive history

        # store initial Fisher
        self.Fisher = self.get_Fisher_func(self.x0, _Lambda)
        self.vals, self.vecs = jnp.linalg.eigh(self.Fisher)
        self.cond_num = jnp.linalg.cond(self.Fisher)

        # vectorize Fisher jump across chains
        self.fast_Fisher_jump = jit(self.Fisher_jump)
        self.vectorized_Fisher_jump = jit(vmap(self.fast_Fisher_jump, in_axes=(0, None, 0, 0, None)))

    
    # jump along eigenvectors of Fisher
    def Fisher_jump(self, state, iteration, temperature, key, history):
        keys = jr.split(key, 2)
        # select direction to jump along
        direction = jr.choice(keys[0], state.shape[0])

        # jump along eigenvector of Fisher scaled by (positive) eigenvalue
        jump = 1. / jnp.sqrt(jnp.abs(self.vals[direction])) * self.vecs[:, direction]
        jump *= jr.normal(keys[1]) * jnp.sqrt(temperature)

        return state + jump
    
    

# Differential evolution
class DifferentialEvolution:

    def __init__(self, len_history):
        self.len_history = len_history  # how many samples in adaptive history
        self.x_min = wg.x_mins
        self.x_max = wg.x_maxs
        self.ndim = wg.ndim
        self.jump_weight = 2.38 / jnp.sqrt(2. * self.ndim)

        # initialize adaptive history
        self.history = jr.uniform(jr.PRNGKey(22), minval=self.x_min, maxval=self.x_max,
                                  shape=(self.len_history, self.ndim))
                                    # ^^ consider adding num_chains as a dimension here so that
                                    # not every chain shares the same history
        
        # vectorize jump over chains
        self.fast_DE_jump = jit(self.DE_jump)
        self.vectorized_DE_jump = jit(vmap(self.fast_DE_jump, in_axes=(0, None, 0, 0, None)))        
 

    def DE_jump(self, state, iteration, temperature, key, history):
        # split random keys
        # draw1_key, draw2_key, weight_key, epsilon_key, replacement_key = jr.split(key, 5)
        draw1_key, draw2_key, weight_key, epsilon_key = jr.split(key, 4)

        # get jump
        jump = jr.choice(draw1_key, history) - jr.choice(draw2_key, history)
        jump *= jr.normal(weight_key) * self.jump_weight
        jump += jr.normal(epsilon_key, shape=(self.ndim,)) * 1.e-4
        # move to new state
        new_state = state + jump

        # update history (DOING IN PTMCMC.py INSTEAD)
        # self.history = self.history.at[jr.choice(replacement_key, self.len_history)].set(jnp.copy(state))

        return new_state

# Prior draw
class PriorDraw:

    def __init__(self):
        # no need to initialize prior (it's represented by jr.uniform below)
        self.history = None # no adaptive history
        # Define jump function
        self.fast_prior_draw = jit(self.prior_draw)
        # vectorize jump over chains
        self.vectorized_prior_draw = jit(vmap(self.fast_prior_draw, in_axes=(0, None, 0, 0, None)))
        
    def prior_draw(self, state, iteration, temperature, key, history): 
        # Needs not be temperature-dependent, because hot (cold) chains will naturally sometimes accept (always reject) this jump type according to chain swap Hastings ratio
        # get, then return, random state from prior (which is just a uniform dist btwn x_mins and x_maxs)
        new_state = jr.uniform(key, minval=wg.x_mins, maxval=wg.x_maxs, shape=state.shape)
        return new_state