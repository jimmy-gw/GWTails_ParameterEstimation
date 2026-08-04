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


# (DEPRECATED) compute Fisher information matrix at given parameter values
# directly evaluate (h_{,i} | h_{,j})
def get_Fisher(x, _Lambda=0):
    Fisher = np.zeros((wg.ndim, wg.ndim))
    for i in range(wg.ndim):
        partial_waveform1 = wg.partial_FD_waveform(x, i, _Lambda)
        for j in range(i, wg.ndim):
            partial_waveform2 = wg.partial_FD_waveform(x, j, _Lambda)
            Fisher[i, j] = Fisher[j, i] = inner(partial_waveform1, partial_waveform2)

    #print(f'Fisher (normal):{Fisher}')
    #assert np.linalg.det(Fisher)!=0, f"Fisher={Fisher} must have nonzero determinant."
    #assert np.all(np.linalg.eigvals(Fisher)>0), f"Fisher={Fisher} must have positive eigenvalues {np.linalg.eigvals(Fisher)}"

    return Fisher


# compute Fisher information matrix at given parameter values
# evaluate Fisher elements using Eq. 9 of https://arxiv.org/pdf/1007.4820
def get_fastFisher(x, _Lambda=0):

    # Define quantities that are reusable for each partial_ampphase call
    h = wg.get_h22(x, _Lambda)
    phi = h.phase

    # Partial derivative (via central finite differencing) helper fxn 
    # (used for Fisher evaluation below)
    def partial_ampphase(mode, x, deriv_ndx, epsilon=1e-8):
        assert mode in ['A','phi'], f"Partial must be specified to be evaluated on AmpPhaseFDWaveform.amp (\"A\") or AmpPhaseFDWaveform.phase (\"phi\")."

        d_kh=wg.partial_FD_waveform(x, deriv_ndx, _Lambda, epsilon)
        if mode=='A':
            return np.real(d_kh * np.exp(-1.j * phi))
        if mode=='phi':
            return np.imag(d_kh * np.exp(-1.j * phi)) # = A*\phi_{,k}, but the A carries twice into the A**2\phi_{,i}\phi_{,j} term in (h_{,j} | h_{,j})

    Fisher = np.zeros((wg.ndim, wg.ndim))
    for i in range(wg.ndim):
        d_iA, Ad_iphi = partial_ampphase('A',x,i), partial_ampphase('phi',x,i)
        for j in range(i,wg.ndim):
            d_jA, Ad_jphi = partial_ampphase('A',x,j), partial_ampphase('phi',x,j)
            # discrete integration
            integrand = (d_iA * d_jA + Ad_iphi * Ad_jphi) / S
            Fisher[i,j] = Fisher[j,i] = 4. * np.sum(integrand) * wg.df

    # Sanity checks
    #print(f'Fisher (fast):{Fisher}')
    #assert np.linalg.det(Fisher)!=0, f"Fisher={Fisher} must have nonzero determinant."
    #assert np.all(np.linalg.eigvals(Fisher)>0), f"Fisher={Fisher} must have positive eigenvalues {np.linalg.eigvals(Fisher)}"

    return Fisher


# Correlation matrix
def get_correlation_matrix(fisher):
    corr = np.zeros((wg.ndim, wg.ndim))
    for i in range(wg.ndim):
         for j in range(i,wg.ndim):
             corr[i,j] = corr[j,i] = fisher[i,j]/np.sqrt(fisher[i,i]*fisher[j,j])
    return corr

# Covariance matrix
def get_cov(fisher):
    return np.linalg.inv(fisher)



# uniform prior
def ln_prior(x):
    out_of_bounds = jnp.logical_or(jnp.any(x < wg.x_mins),
                                   jnp.any(x > wg.x_maxs))
    def out_of_bounds_case():
        return -jnp.inf
    def in_bounds_case():
        return 0.0
    return jax.lax.cond(out_of_bounds, out_of_bounds_case, in_bounds_case)

fast_lnprior = jax.jit(ln_prior)


# likelihood
# NB: The local scope of the lambdas here differs from the global scope of these same 
# variables that, in the rest of the repo, are carried through the data_amp/phase_suppressed objects from data.py.

##################################################################################################################################################
# Code quarantine: the integrands' dependence on data.py vars might amount to
# something different than what is needed in PTMCMC. I think ultimately they
# have to be noise weighted inner products (d-h|d-h) where d (or maybe h? idk I'm brainfried)
# is totally unsuppressed
def ln_likelihood(x, temperature=1.0, _Lambda=0):
    x_h22 = wg.get_h22(x, _Lambda)
    x_amp, x_phase = x_h22.amp, x_h22.phase
    integrand = (x_amp**2 + d.data_amp**2 - 2 * x_amp * d.data_amp * np.cos(x_phase - d.data_phase)) / S
    lnlike = -2. * np.sum(integrand) * wg.df
    return lnlike / temperature

# cut off calculation of these log-likelihood functions at f_{IM} independently of waveform

def loglike_untempered(x, _Lambda=0):
    x_h22 = wg.get_h22(x, _Lambda)
    x_amp, x_phase = x_h22.amp, x_h22.phase
    integrand = (x_amp**2 + d.data_amp**2 - 2 * x_amp * d.data_amp * np.cos(x_phase - d.data_phase)) / S
    lnlike = -2. * np.sum(integrand) * wg.df
    return lnlike

##################################################################################################################################################



# posterior
def ln_posterior(x, temperature=1.0, _Lambda=0):
    if np.any(x < wg.x_mins) or np.any(x > wg.x_maxs):
        return -np.inf
    else:
        return ln_likelihood(x, temperature, _Lambda)

def ln_posterior_suppressed(x, temperature=1.0, _Lambda=1):
    if np.any(x < wg.x_mins) or np.any(x > wg.x_maxs):
        return -np.inf
    else:
        return ln_likelihood(x, temperature, _Lambda)







