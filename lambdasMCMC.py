'''MCMC analysis for parameter estimation of toggling parameters lambda15, lambda25, lambda3, and lambda35[?].'''



import data as d
import wave_gen as wg
import numpy as np
import jumps as j
from PTMCMC import PTMCMC
import matplotlib.pyplot as plt
import likelihood as l
import seaborn as sns
#from numba import njit


#******************************************************** ********************************************************
#******************** Initialization ******************** ********************************************************
#******************************************************** ********************************************************

# Initialize pure unsuppressed waveform
data_h22_0=d.data_h22
data_amp_0, data_phase_0 = np.array(data_h22_0.amp), np.array(data_h22_0.phase)
data_FD_waveform_0 = d.data_FD_waveform

# Initialize suppression MCMC parameters

###############################################
# Figure out how to cheapen the cost of doing PTMCMCs before increasing this number (currently set to 3 for testing purposes) 
# maybe find right placement for @njit decorators? Learn how jax.jit() works too
num_iters = 2
###############################################

lambda15s = np.zeros(num_iters)
lambda25s = np.zeros(num_iters)
lambda3s = np.zeros(num_iters)
lambda35s = np.zeros(num_iters)
itercolors = sns.color_palette("Dark2", n_colors=num_iters)

# 6/9/26: While configuring the PT swaps for 10-20% acceptance rates based on lnL stdev overlap, don't apply any suppression to injected data

for i in range(num_iters):
    lambda15s[i] = i#np.random.uniform(0,1)
#    lambda25s[i] = np.random.uniform(0,1)
#    lambda3s[i] = np.random.uniform(0,1)
#    lambda35s[i] = np.random.uniform(0,1)

#******************************************************** ********************************************************
#************************ PTMCMC ************************ ********************************************************
#******************************************************** ********************************************************

# do PTMCMC!
def PTMCMC_i(lambdas: list, num_samples=10000, num_chains=15, temp0=1.3, return_acceptance_rates=True): # 6/3/26: num_samples is taken to be 10K instead of 100K for now... this code might be wicked expensive
    '''Run PTMCMC for given lambda values, taken from the loop over the zipped lambda lists.'''
    lambda15, lambda25, lambda3, lambda35 = lambdas

    # jump proposals
    Fisher = j.Fisher(d.x_inj, lambda15, lambda25, lambda3, lambda35)
    diff_evol = j.DifferentialEvolution(len_history=100)
    jump_proposals = [[Fisher.vectorized_Fisher_jump, 20],
                    [diff_evol.vectorized_DE_jump, 20]]

    samples, lnposts, temp_ladder, acc_rates = PTMCMC(num_samples=num_samples,
                                        num_chains=num_chains,
                                        x0=d.x_inj,
                                        ln_posterior_func=l.ln_posterior,
                                        jump_proposals=jump_proposals,
                                        PT_swap_weight=10,
                                        lambda15=lambda15, lambda25=lambda25, lambda3=lambda3, lambda35=lambda35,
                                        temp0=temp0,
                                        return_acceptance_rates=return_acceptance_rates)
    
    return samples, lnposts, temp_ladder, acc_rates


#******************************************************** ********************************************************
#*********************** Evidence *********************** ********************************************************
#******************************************************** ********************************************************


### GET RELIABLE ERROR BARS ON THE MEAN LOG LIKELIHOODS BEFORE USING THIS FUNCTION
# Calculate log evidence from meanlnlikes vs beta plot using thermodynamic integration
def lnevidence_TI(lnposts, temp_ladder):

    # code-in a return statement for avglnlikes_vs_betas()?

    # Use trapezoidal rule to integrate <logL> over beta
#    lnZ = np.trapz(avg_lnlikes, betas)
#    return lnZ
    pass

def bayesFactor(lnZ1, lnZ2):
    '''Calculate Bayes factor from two sets of log evidences.'''
    return lnZ2 - lnZ1


#******************************************************** ********************************************************
#******************* Helper functions ******************* ***************** (Plotting functions) *****************
#******************************************************** ********************************************************

# Unnecessary functions for evidence calculations
def phasediff_vs_freq(lambdas1=[0,0,0,0], lambdas2=[d.lambda15,d.lambda25,d.lambda3,d.lambda35]):
    '''Calculate and plot (as a fxn of frequency) the phase difference \Psi_2-\Psi1 between two waveforms h22_2 and h22_1, 
    with the same injected parameters but differing degrees of suppression. Suppression for each waveform is controlled 
    through lambdas1 and lambdas2, each a list of lambda15 thru lambda35.'''
    h22_1 = wg.get_h22(d.x_inj, lambdas1[0], lambdas1[1], lambdas1[2], lambdas1[3])
    h22_2 = wg.get_h22(d.x_inj, lambdas2[0], lambdas2[1], lambdas2[2], lambdas2[3])
    delta_psi = h22_2.phase - h22_1.phase
    
    plt.plot(wg.f,delta_psi)
    plt.xlabel('frequency [Hz]')
    plt.ylabel(r'$\Delta\Psi(f)=\Psi_{2}(f)-\Psi_{1}(f)$')
    plt.title('lambdas1={}, lambdas2={}'.format(lambdas1, lambdas2))
    plt.show()


def chain_heatmap(temp_ladder, lnposts, num_chains):
    '''Plots likelihood values of chains (without temperature scaling).
        temp_ladder: An output of PTMCMC().
        lnposts: An output of PTMCMC().
        num_chains: An input to PTMCMC().'''
    chain_colors = list(reversed([plt.cm.plasma(i / num_chains) for i in range(1, num_chains + 1)]))

    for j, (temp, color) in enumerate(zip(temp_ladder[::-1], chain_colors)):
        plt.plot(lnposts[::-1][j] * temp, color=color, label=f'T = {round(temp, 3)}')
    plt.title('PTMCMC lambdavec=[nothing,here,asof,yet]') # figure out how to include toggle parameter values in title
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
    plt.title('PTMCMC lambdavec=[alsonothing,here,asof,yet]') # figure out how to include toggle parameter values in title
    plt.xlabel('MCMC iteration')
    plt.ylabel('parameter value')
    plt.legend(loc='lower right')
    plt.show()

# Necessary function for evidence calculation
def avglnlikes_vs_betas(temp_ladder, lnposts, num_samples): # pass in a figure argument to paste in a plot, or better yet return the betas and avg_lnlikes
    '''returns average log likelihoods vs betas, for each chain, for ONE set of suppression parameters.'''
    burnin = num_samples // 5
    avg_lnlikes = np.mean(lnposts[:, burnin:], axis=1) * temp_ladder
    lnlikes_stdevs = np.std(lnposts[:, burnin:], axis=1) * temp_ladder
    betas = 1. / temp_ladder

    return betas, avg_lnlikes, lnlikes_stdevs

def ladder_spacing(temp_ladder, avg_lnlikes, lnlikes_stdevs, alpha=1, print_chains=True, haxis='T'):
    '''Returns R=\delta<lnL> / alpha(sigma_i+sigma_i+1) versus beta, where \delta refers to the difference between a chain and its hotter neighbor.
    R is used to determine the fidelity of the temperature ladder.'''
    # I might just tweak alpha manually but I know I can calculate it analytically too. Maybe build a separate helper function?

    if haxis!='T' and haxis!='B':
        return ValueError('haxis must be equal to "T" or "B".')

    R=np.zeros(len(list(temp_ladder)))

    for j in range(len(temp_ladder)-1):
        dmeanlnL=avg_lnlikes[j+1]-avg_lnlikes[j]
        ddmeanlnL = None # error in the mean. Incorporate somehow w/ formula
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
