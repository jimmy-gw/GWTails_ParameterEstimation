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

def trace_temp1chain(samples) # samples may need to be carried over from PTMCMC_i()
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
def avglnlikes_vs_betas(temp_ladder, lnposts, num_samples, color):
    '''Plots average log likelihoods vs betas, for each chain, for ONE set of suppression parameters.
        color: Taken from global variable itercolors, which is a list of colors for each iteration of lambdaMCMC.'''
    burnin = num_samples // 5
    avg_lnlikes = np.mean(lnposts[:, burnin:], axis=1) * temp_ladder
    betas = 1. / temp_ladder

    plt.plot(betas, avg_lnlikes, marker='.', color=color)
    plt.xlabel(r'$\beta = 1 / T$')
    plt.ylabel(r'$\langle\log\mathcal{L}\rangle$')

# call plt.legend() and plt.show() outside of this function (avglnlikes_vs_betas) so that it displays all iterations of lambdaMCMC at once


#******************************************************** ********************************************************
#******************** Initialization ******************** ********************************************************
#******************************************************** ********************************************************

# Initialize pure unsuppressed waveform
data_h22_0=d.data_h22
data_amp_0, data_phase_0 = np.array(data_h22_0.amp), np.array(data_h22_0.phase)
data_FD_waveform_0 = d.data_FD_waveform

# Initialize suppression MCMC parameters
num_iters = 3 # Figure out how to cheapen the cost of doing PTMCMCs (find right placement for @njit decorators?) 
# before increasing this number (currently set to 3 for testing purposes)

lambda15s = np.zeros(num_iters)
lambda25s = np.zeros(num_iters)
lambda3s = np.zeros(num_iters)
lambda35s = np.zeros(num_iters)
itercolors = sns.color_palette("Dark2", n_colors=num_iters)

for i in range(num_iters):
    lambda15s[i] = np.random.uniform(0,1)
    lambda25s[i] = np.random.uniform(0,1)
    lambda3s[i] = np.random.uniform(0,1)
    lambda35s[i] = np.random.uniform(0,1)

    #randsupp_h22 = wg.get_h22(d.x_inj, lambda25s[i], lambda3s[i])
    #randsupp_amp, randsupp_phase = np.array(randsupp_h22.amp), np.array(randsupp_h22.phase)


#******************************************************** ********************************************************
#************************* PTMCMC ************************* ********************************************************
#******************************************************** ********************************************************

# jump proposals
def jump_proposal_Fisher():
    Fisher = j.Fisher(d.x_inj)
    diff_evol = j.DifferentialEvolution(len_history=100)
    jump_proposals = [[Fisher.vectorized_Fisher_jump, 20],
                    [diff_evol.vectorized_DE_jump, 20]]
    return Fisher, diff_evol, jump_proposals

# do PTMCMC!
def PTMCMC_i(lambdas: list, num_samples=10000, num_chains=15): # 6/3/26: num_samples is taken to be 10K instead of 100K for now... this code might be wicked expensive
    '''Run PTMCMC for given lambda values, taken from the loop over the zipped lambda lists.'''
    lambda15, lambda25, lambda3, lambda35 = lambdas
    samples, lnposts, temp_ladder = PTMCMC(num_samples=num_samples,
                                        num_chains=num_chains,
                                        x0=d.x_inj,
                                        ln_posterior_func=l.ln_posterior,
                                        jump_proposals=jump_proposals,
                                        PT_swap_weight=10,
                                        lambda15, lambda25, lambda3, lambda35)
    return samples, lnposts, temp_ladder


#******************************************************** ********************************************************
#*********************** Evidence *********************** ********************************************************
#******************************************************** ********************************************************


### GET RELIABLE ERROR BARS ON THE MEAN LOG LIKELIHOODS BEFORE USING THIS FUNCTION
# Calculate log evidence from meanlnlikes vs beta plot using thermodynamic integration
def lnevidence_TI(lnposts, temp_ladder):

    # code-in a return statement for avglnlikes_vs_betas()?

    # Use trapezoidal rule to integrate <logL> over beta
    lnZ = np.trapz(avg_lnlikes, betas)
    return lnZ

def bayesFactor(lnZ1, lnZ2):
    '''Calculate Bayes factor from two sets of log evidences.'''
    return lnZ2 - lnZ1