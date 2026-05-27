'''Script contains prior, likelihood, and posterior densities.
Also contained are functions for the noise weighted inner product and Fisher matrix.'''


import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d
import wave_gen as wg
import data as d


# use double precision
jax.config.update("jax_enable_x64", True)


# load theoretical sensitivity curve from aLIGO
### (May need to update this; cf. Aiden's email)
aLIGO_sensitivity = np.loadtxt('aLIGODesign.txt')
# interpolate over our frequency bins
sqrtS = jnp.array(interp1d(aLIGO_sensitivity[:, 0], aLIGO_sensitivity[:, 1])(wg.f))
S = sqrtS**2.


# noise weighted inner product between two frequency-domain waveforms
# input complex amplitude as array for each waveform
def inner(a, b):
    integrand = (jnp.real(a) * jnp.real(b) + jnp.imag(a) * jnp.imag(b)) / S
    return 4. * jnp.sum(integrand) * wg.df

fast_inner = jax.jit(inner)


# compute Fisher information matrix at given parameter values
def get_Fisher(x):
    Fisher = np.zeros((wg.ndim, wg.ndim))
    for i in range(wg.ndim):
        partial_waveform1 = wg.partial_FD_waveform(x, i)
        for j in range(i, wg.ndim):
            partial_waveform2 = wg.partial_FD_waveform(x, j)
            Fisher[i, j] = Fisher[j, i] = inner(partial_waveform1, partial_waveform2)
    return Fisher


# uniform prior
def ln_prior(x, temperature=1.0):
    out_of_bounds = jnp.logical_or(jnp.any(x < wg.x_mins),
                                   jnp.any(x > wg.x_maxs))
    def out_of_bounds_case():
        return -jnp.inf
    def in_bounds_case():
        return 0.0
    return jax.lax.cond(out_of_bounds, out_of_bounds_case, in_bounds_case)

fast_lnprior = jax.jit(ln_prior)


# likelihood
# NB: The local scope of lambda25 and lambda3 conflicts with the global scope of these 
# variables (carried through the data_amp/phase_suppressed objects) in data.py.
def ln_likelihood(x, temperature=1.0, lambda25=0, lambda3=0):
    x_h22 = wg.get_h22(x, lambda25, lambda3)
    x_amp, x_phase = x_h22.amp, x_h22.phase
    if lambda25 == 0 and lambda3 == 0:
        integrand = (x_amp**2 + d.data_amp**2 - 2 * x_amp * d.data_amp * np.cos(x_phase - d.data_phase)) / S
    else:
        integrand = (x_amp**2 + d.data_amp_suppressed**2 - 2 * x_amp * d.data_amp_suppressed * np.cos(x_phase - d.data_phase_suppressed)) / S
    lnlike = -2. * np.sum(integrand) * wg.df
    return lnlike / temperature


# posterior
def ln_posterior(x, temperature=1.0, lambda25=0, lambda3=0):
    if np.any(x < wg.x_mins) or np.any(x > wg.x_maxs):
        return -np.inf
    else:
        return ln_likelihood(x, temperature, lambda25, lambda3)







