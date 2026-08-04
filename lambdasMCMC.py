'''MCMC analysis for parameter estimation of suppression parameter _Lambda'''


import data as d
import wave_gen as wg
import numpy as np
import jumps as j
from PTMCMC import PTMCMC
import matplotlib.pyplot as plt
import likelihood as l
import seaborn as sns
from scipy import special
from scipy.interpolate import Akima1DInterpolator
import IMRPhenomD.IMRPhenomD_const

# corner plot helper packages
import matplotlib.patches as mpatches
from corner import corner
from scipy.stats import multivariate_normal


#from numba import jit


#******************************************************** ********************************************************
#******************** Initialization ******************** ********************************************************
#******************************************************** ********************************************************


# Initialize pure unsuppressed waveform
data_h22_0=d.data_h22
data_amp_0, data_phase_0 = np.array(data_h22_0.amp), np.array(data_h22_0.phase)
data_FD_waveform_0 = d.data_FD_waveform



Lambdas = np.array([0,1])
num_iters = np.shape(Lambdas)[0]
itercolors = sns.color_palette("Dark2", n_colors=num_iters)

#******************************************************** ********************************************************
#************************ PTMCMC ************************ ********************************************************
#******************************************************** ********************************************************


# do PTMCMC!
def PTMCMC_i(_Lambda, temp_ladder, num_samples=10000, num_chains=15, return_acceptance_rates=True):
    '''Run PTMCMC for given lambda values, taken from the loop over the zipped lambda lists.'''

    # jump proposals
    Fisher = j.Fisher(d.x_inj, _Lambda)
    diff_evol = j.DifferentialEvolution(len_history=100)
    prior_draw = j.PriorDraw()
    jump_proposals = [[Fisher.vectorized_Fisher_jump, 20],
                    [diff_evol.vectorized_DE_jump, 20],
                    [prior_draw.vectorized_prior_draw, 0.91370559]] # << 1.5% chance of prior draw (given weight=20 for other 3 jump types)

    samples, lnposts, acc_rates = PTMCMC(num_samples=num_samples,
                                        num_chains=num_chains,
                                        x0=d.x_inj,
                                        ln_posterior_func=l.ln_posterior,
                                        jump_proposals=jump_proposals,
                                        temp_ladder=temp_ladder,
                                        PT_swap_weight=20,
                                        _Lambda=_Lambda,
                                        return_acceptance_rates=return_acceptance_rates)
    
    return samples, lnposts, acc_rates


#******************************************************** ********************************************************
#********** Adaptive temperature ladder spacing ********* ********************************************************
#******************************************************** ********************************************************



# threw all my old stochastic adaptation stuff into the junkyard -_-



#******************************************************** ********************************************************
#*********************** Evidence *********************** ********************************************************
#******************************************************** ********************************************************

class TIData:

    def __init__(self, betas, lnlikes, avg_lnlikes, lnlikes_stdevs, _Lambda, color, filenum):
        self.betas=betas
        self.lnlikes=lnlikes # want this to be a 2D array where each column denotes a beta value with its list of num_samples lnlikes
        self.avg_lnlikes=avg_lnlikes
        self.lnlikes_stdevs=lnlikes_stdevs
        self._Lambda=_Lambda
        self.color=color
        self.filenum=filenum

        self.evidence=None

    def generatePerturbedLogLikesToSpline(self):
        '''Takes a <lnL> vs. beta plot, varies each point by a random amount chosen from a Gaussian distribution centered on 0 with 1 sigma = size of error bar, and then returns it. The output is to be Akima-splined.'''
        avg_lnlikes_perturbed=self.avg_lnlikes+np.array([np.random.normal(0,sigma) for sigma in self.lnlikes_stdevs])
        return avg_lnlikes_perturbed

def bayesFactor(lnZ1, lnZ2):
    '''Calculate Bayes factor from two sets of log evidences.'''
    return lnZ2 - lnZ1

def lnlikes(temp_ladder, lnposts, num_samples):
    '''returns log likelihoods for each chain, for ONE set of suppression parameters.'''
    burnin = num_samples // 5
    lnlikes = np.array(lnposts[:, burnin:]) * temp_ladder[:,None] # lnposts is not collapsed along axis=1 as it is when taking mean or std, hence the [:,None]
    return np.transpose(lnlikes)

def construct_te_dattab(betas, lnlikes):
    '''Creates a 2D np.array object akin to logLchain.dat, and saves it as a .dat file to be read by thermo_error.py.
        The first column is a counter list: top element is 0, all others are 1. Size=num_samples+1.
        The first row (after the 0 in the top left) is betas, in descending order from 1 to ~0.
        All columns under each beta are num_samples number of lnL samples.'''
    
    arr = np.empty((lnlikes.shape[0] + 1, lnlikes.shape[1] + 1))
    arr[0, 0] = 0
    arr[0, 1:] = betas
    arr[1:, 0] = np.full(lnlikes.shape[0],1)
    arr[1:, 1:] = lnlikes

    # call np.savetxt(filename, arr, fmt='%.12g') outside
    return arr



#******************************************************** ********************************************************
#******************* Helper functions ******************* ****************** Plotting functions ******************
#******************************************************** ********************************************************


class PTSamples:
    def __init__(self, samples, _Lambda):
        # Basic attributes
        self.samples=samples
        self._Lambda=_Lambda

        # Dependent quantities
        self.Ms=self.get_Ms(self.samples)                            # shape: (num_chains, num_samples)
        self.fIMs=IMRPhenomD.IMRPhenomD_const.PHI_fJoin_INS/self.Ms  # shape: (num_chains, num_samples)
        self.thSNRs=self.get_thSNRs(self.samples)
        # Chirp mass?
        # Spin variables s_l, sigma_l, \chi_s, \chi_a?

    def get_Ms(self, samps):
        '''Return total mass by sample'''
        m1_samples = samps[:, :, 0]   # shape: (num_chains, num_samples)
        m2_samples = samps[:, :, 1]   # shape: (num_chains, num_samples)
        M_samples=m1_samples+m2_samples
        return M_samples

    def get_DQsamples(self):
        '''Returns samples of dependent quantities (DQs) with shape (num_chains, num_samples, {num_DQs}).'''
        return self.Ms, self.fIMs, self.thSNRs

    def get_thSNRs(self, samps):
        '''Returns theoretical SNR (h|h).'''


        # Sanity check on computing time
        print('Computing SNRs...')



        # samps: (num_chains, num_samples, n_dim)
        num_chains, num_samples, ndim = samps.shape

        # flatten to one parameter vector per row
        x_flat = samps.reshape(-1, ndim) # (num_chains*num_samples, ndim)

        # map FD_waveform over every parameter vector
        h_flat = np.array([wg.FD_waveform(x, self._Lambda) for x in x_flat])

        # reshape back to (num_chains, num_samples, Nf)
        h_stack = h_flat.reshape(num_chains, num_samples, -1)

        # compute (h|h) per sample
        snr = np.array([l.fast_inner(h, h) for h in h_stack.reshape(-1, h_stack.shape[-1])])


        # Sanity check on computing time
        print('Computing SNRs...')



        return snr.reshape(num_chains, num_samples)


# vv Unnecessary for evidence calculations
def phasediff_vs_freq(lambda1=0, lambda2=[d._Lambda]):
    '''Calculate and plot (as a fxn of frequency) the phase difference \Psi_2-\Psi1 between two waveforms h22_2 and h22_1, 
    with the same injected parameters but differing degrees of suppression. Suppression for each waveform is controlled 
    through lambdas1 and lambdas2, each a list of lambda15 thru lambda35.'''
    h22_1 = wg.get_h22(d.x_inj, lambda1)
    h22_2 = wg.get_h22(d.x_inj, lambda2)
    delta_psi = h22_2.phase - h22_1.phase
    
    plt.plot(wg.f,delta_psi)
    plt.xlabel('frequency [Hz]')
    plt.ylabel(r'$\Delta\Psi(f)=\Psi_{2}(f)-\Psi_{1}(f)$')
    plt.title(rf'$\lambda_1$={lambda1}, $\lambda_2$={lambda2}')
    plt.show()


def chain_heatmap(temp_ladder, lnposts, num_chains):
    '''Plots likelihood values of chains (without temperature scaling).
        temp_ladder: An output of PTMCMC().
        lnposts: An output of PTMCMC().
        num_chains: An input to PTMCMC().'''
    chain_colors = list(reversed([plt.cm.plasma(i / num_chains) for i in range(1, num_chains + 1)]))

    for j, (temp, color) in enumerate(zip(temp_ladder[::-1], chain_colors)):
        plt.plot(lnposts[::-1][j] * temp, color=color, label=f'T = {round(temp, 3)}')
    plt.title('PTMCMC lambda=[nothinghereasofyet]') # figure out how to include toggle parameter values in title
    plt.xlabel('MCMC iteration')
    plt.ylabel('log(likelihood)')
    plt.legend(loc='lower right')
    plt.show()

def trace_temp1chain(samples):
    '''Creates a trace plot for T = 1 chain.
        samples: An output of PTMCMC().'''
    for i in range(wg.ndim):
        plt.plot(samples[0,:,i], color=f'C{i}', alpha=0.6)
        plt.axhline(d.x_inj[i], color=f'C{i}', alpha=0.8, label=wg.x_labels[i])
    plt.title('PTMCMC lambda=[alsonothinghereasofyet]') # figure out how to include toggle parameter values in title
    plt.xlabel('MCMC iteration')
    plt.ylabel('parameter value')
    plt.legend(loc='lower right')
    plt.show()

def cornerplots(samples, _Lambda=0):
    '''Creates corner plots displaying the results of PTMCMC sampling, as well as Fisher samples.'''
    # for reference, samples is the first output of PTMCMC()
    num_samples=np.shape(samples)[1]
    burnin=num_samples//5

    # PTMCMC samples
    pt_samples = samples[0, burnin:]
    pt_n = pt_samples.shape[0]
    pt_weights = np.full(pt_n, 1.0 / pt_n)

    x0=d.x_inj
    labels=wg.x_labels


    # Fisher samples
    Fisher = j.Fisher(d.x_inj, _Lambda)
    cov = np.linalg.inv(Fisher.Fisher)
    Fisher_samples = multivariate_normal(d.x_inj, cov, allow_singular=True).rvs(num_samples - burnin)
    fisher_n = Fisher_samples.shape[0]
    fisher_weights = np.full(fisher_n, 1.0 / fisher_n)

    fig = corner(
        Fisher_samples,
        truths=x0,
        labels=labels,
        color='blue',
        bins=30,
        hist_kwargs={'density': True},
        weights=fisher_weights,
    )

    corner(
        pt_samples,
        color='red',
        fig=fig,
        bins=30,
        hist_kwargs={'density': True},
        weights=pt_weights,
    )


    red_patch = mpatches.Patch(color='red', label='PTMCMC samples')
    blue_patch = mpatches.Patch(color='blue', label='Fisher draws')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles += [red_patch, blue_patch]
    fig.legend(handles=handles, loc='upper right', fontsize=24)
    plt.show()

# ^^ Unnecessary for evidence calculations


def avglnlikes_vs_betas(temp_ladder, lnposts, num_samples):
    '''returns average log likelihoods vs betas, for each chain, for ONE set of suppression parameters.'''
    burnin = num_samples // 5
    avg_lnlikes = np.mean(lnposts[:, burnin:], axis=1) * temp_ladder # this holds all num_samples lnlikes
    lnlikes_stdevs = np.std(lnposts[:, burnin:], axis=1) * temp_ladder
    betas = 1. / temp_ladder

    return betas, avg_lnlikes, lnlikes_stdevs

def alpha_overlap(avg_lnlike_1, lnlikes_stdev_1, avg_lnlike_2, lnlikes_stdev_2, order=1):
    '''Calculates overlap factor alpha according to Eq. (12) of Rathmore et al. 2004 [DOI: 10.1063/1.1831273].
    avg_lnlike_2 and lnlike_stdev_2 are assumed greater than that of 1.'''

    # (!!!) Before running this... how do the distributions of lnlikes get normalized?'

    if order not in [1,2]:
        raise ValueError(f'order={order} must be 1 or 2, denoting 1st- or 2nd-order asymptotic expansion in (\sigma_2/\sigma_1-1)')

    if avg_lnlike_1>avg_lnlike_2: # avg_lnlike_2 is strictly larger than avg_lnlike_1, so swap them if this isn't the case
        mucopy=avg_lnlike_1
        avg_lnlike_1=avg_lnlike_2
        avg_lnlike_2=mucopy

        sigcopy=lnlikes_stdev_1
        lnlikes_stdev_1=lnlikes_stdev_2
        lnlikes_stdev_2=sigcopy
    
    dmu = avg_lnlike_2-avg_lnlike_1
    sigm = (lnlikes_stdev_1+lnlikes_stdev_2) / 2.
    z = dmu / sigm
    leadTerm = special.erfc(z / (np.sqrt(8.)))

    if order==1:
        correxn = 0
    elif order==2:
        correxn = (-np.exp(-z**2 / 2) * (lnlikes_stdev_2/lnlikes_stdev_1-1)**2) / (np.pi * np.sqrt(8) * z) #I've set 'g' as it appears in eq. 13 to 2 just so I can write something complete. 
        #^^ This correction is probably negligible for my purposes anyways. Check!

    return leadTerm + correxn

def ladder_spacing_ratio(temp_ladder, avg_lnlikes, lnlikes_stdevs, alphas, print_chains=True, haxis='T'):
    '''Returns R=\delta<lnL> / alpha(sigma_i+sigma_i+1) versus beta, where \delta refers to the difference between a chain and its hotter neighbor.
    R is used to determine the fidelity of the temperature ladder.'''
    # I might just tweak alpha manually but I know I can calculate it analytically too. Maybe build a separate helper function?

    if haxis!='T' and haxis!='B':
        return ValueError('haxis must be equal to "T" or "B".')

    R=np.zeros(len(list(temp_ladder)))

    for j in range(len(temp_ladder)-1):
        dmeanlnL=avg_lnlikes[j+1]-avg_lnlikes[j]
        ddmeanlnL = None # error in the mean. Incorporate somehow w/ formula
        alpha=alphas[j]
        reach=alpha*(lnlikes_stdevs[j]+lnlikes_stdevs[j+1])
        overlap = (reach>np.abs(dmeanlnL)) # do the errorbars overlap with one another up to a factor alpha
        R[j]=np.abs(dmeanlnL)/reach

        if print_chains:
            temp=temp_ladder[j]
            print(f'Chain {j+1} (T={round(temp,3)}):\n \t<lnL>_{j+2}-<lnL>_{j+1}={dmeanlnL},\n \talpha(sigma_{j+1}+sigma_{j+2})={reach},\n \toverlap w/ chain {j+2}? {overlap}')

    if haxis=='T':
        return temp_ladder, R
    if haxis=='B':
        betas=1./temp_ladder
        return betas, R