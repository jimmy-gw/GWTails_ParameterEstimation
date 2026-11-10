'''MCMC analysis for parameter estimation of suppression parameters _Alpha5Ln, _Alpha6Ln'''


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
import IMRPhenomD.IMRPhenomD_const as imrc

# corner plot helper packages
import matplotlib.patches as mpatches
from corner import corner
from scipy.stats import multivariate_normal

# from jax import jit, vmap
# from numba import jit


#******************************************************** ********************************************************
#******************** Initialization ******************** ********************************************************
#******************************************************** ********************************************************


# Initialize pure unsuppressed waveform
data_h22_0=d.data_h22
data_amp_0, data_phase_0 = np.array(data_h22_0.amp), np.array(data_h22_0.phase)
data_FD_waveform_0 = d.data_FD_waveform



Alphas = np.array([(1,1),(1,0)]) # no suppression, tail-of-tail suppression only
num_iters = np.shape(Alphas)[0]
itercolors = sns.color_palette("Dark2", n_colors=num_iters)

#******************************************************** ********************************************************
#************************ PTMCMC ************************ ********************************************************
#******************************************************** ********************************************************


# do PTMCMC!
def PTMCMC_i(_Alpha5Ln,_Alpha6Ln, temp_ladder, num_samples, num_chains, return_acceptance_rates=True):
    '''Run PTMCMC for given alpha values, taken from the loop over the zipped alpha lists.'''

    # jump proposals
    Fisher = j.Fisher(d.x_inj, _Alpha5Ln,_Alpha6Ln)

    diff_evol = j.DifferentialEvolution(len_history=100)
    prior_draw = j.PriorDraw()

    # maybe rewrite these lists as [class, vectorized jump, weight], and edit PTMCMC() accordingly?
    jump_proposals = [[Fisher, Fisher.vectorized_Fisher_jump, 20],
                    [diff_evol, diff_evol.vectorized_DE_jump, 20],
                    [prior_draw, prior_draw.vectorized_prior_draw, 0.30456853]] # << 0.5% chance of prior draw (given weight=20 for other 3 jump types)

    samples, lnposts, acc_rates = PTMCMC(num_samples=num_samples,
                                        num_chains=num_chains,
                                        x0=d.x_inj,
                                        ln_posterior_func=l.ln_posterior,
                                        jump_proposals=jump_proposals,
                                        temp_ladder=temp_ladder,
                                        PT_swap_weight=20,
                                        _Alpha5Ln=_Alpha5Ln,
                                        _Alpha6Ln=_Alpha6Ln,
                                        return_acceptance_rates=return_acceptance_rates)
    
    return samples, lnposts, acc_rates


#******************************************************** ********************************************************
#********** Adaptive temperature ladder spacing ********* ********************************************************
#******************************************************** ********************************************************

class TempLadder:

    def __init__(self, temps):
        self.temps = temps
        self.betas = 1./temps
        self.etas = self.set_etas(temps)
        self.gammas = self.set_gammas()

    # Chain density \eta
    def set_etas(self, temp_ladder):
        '''Returns the density of chains eta=\frac{1}{\ln\gamma}, where \gamma is the ratio \frac{T_{i+1}}{T_i}, vs temperature.
            Because \gamma does not exist for Tmax, the last value for eta defaults to 0 (corresponding to T_{i+1}=\infty).'''
    
        etas = np.zeros(shape=temp_ladder.shape[0])
        etas[-1]=0
    
        for i in range(temp_ladder.shape[0]-1):
            j = i+1
            Ti, Tj = temp_ladder[i], temp_ladder[j]
            etas[i] = 1. / np.log10(Tj/Ti)
    
        # Normalize by appending a factor of num_chains / ( \int_0^{\log_{10}T_{max}}\eta d(\log_{10}T)) to etas
        logTs = np.log10(temp_ladder)
        Nc = temp_ladder.shape[0]
        integral = np.trapezoid(etas, logTs, dx=0.1)
        etas *= (Nc/integral)
    
        return etas

    def get_etas(self):
        return self.etas

    # Chain spacings \gamma
    def set_gammas(self):
        return 10**(1./self.etas)

    def get_gammas(self):
        return self.gammas

    # # Residual between eta and sqrtCv
    # def eta_Cv_residuals(self, etas, Cvs):
    #     return etas-Cvs

    # Conversion functions
    def etas_to_gammas(self, etas):
        gammas = 10.**(1./etas)
        return gammas
    def gammas_to_etas(self, gammas):
        etas = 1./np.log10(gammas)
        return etas

    def gammas_to_temps(self, gammas):
        temp_ladder = [1]
        for gamma in gammas:
            temp_ladder.append(temp_ladder[-1]*gamma)
        return np.array(temp_ladder)
    def temps_to_gammas(self, temp_ladder):
        gammas = np.zeros(temp_ladder.shape[0]-1) # Tmax has no associated gamma value
        for i in range(temp_ladder.shape[0]-1):
            j = i+1
            Ti, Tj = temp_ladder[i], temp_ladder[j]
            gammas[i] = Tj/Ti
        return gammas

    def convert_tempsbetas(self, TB):
        '''convert temps (betas) to betas (temps)'''
        return 1./TB 




# Discrete method for calculating specific heats
def Cv(Ts, dlnLs):
    '''returns specific heat C_v=-\frac{\del\ln\mathcal{L}}{\del T}=\beta^2Var(\ln\mathcal{L})
    NOTE peaks in C_v correspond to phase transitions, where the temperature ladder should naturally bunch up.
    '''
    Cvs=(1./Ts**2)*dlnLs**2
    return Cvs

# Continuous (splining) method for calculating specific heats
def Cv_spl_func(Ts, lnLs):
    '''returns an interpolat*OR* for specific heat C_v=-\frac{\del\ln\mathcal{L}}{\del T}} by splining and differentating -\ln L(T).'''
    spliner = Akima1DInterpolator(Ts, -lnLs, method='makima') # interp, -lnLs so that Cv picks up the necessary minus sign
    return spliner.derivative()

def Cv_spl(Ts, lnLs):
    '''returns interpolat*ED* specific heat C_v=-\frac{\del\ln\mathcal{L}}{\del T}}'''
    deriv = Cv_spl_func(Ts, lnLs)

    Ts_spl = np.geomspace(Ts[0],Ts[-1], 100)
    Cvs_spl = deriv(Ts_spl) # minus sign already carried through
    return Ts_spl, Cvs_spl

# Chain density helper
def chaindens_vs_temps(temp_ladder):
    '''Returns the density of chains eta=\frac{1}{\ln\gamma}, where \gamma is the ratio \frac{T_{i+1}}{T_i}, vs temperature.
    Because \gamma does not exist for Tmax, the last value for eta defaults to 0 (corresponding to T_{i+1}=\infty).'''

    etas = np.zeros(shape=temp_ladder.shape[0])
    etas[-1]=0

    for i in range(temp_ladder.shape[0]-1):
        j = i+1
        Ti, Tj = temp_ladder[i], temp_ladder[j]
        etas[i] = 1. / np.log10(Tj/Ti)

    # Normalize so that integral of etas is num_chains

    return etas

# Helper to derive chain structure from root specific heat
def copy_etas_from_Cvs(num_chains, scaling, Cv_func: Akima1DInterpolator):
    '''Using a spline-interpolated function of arbitrary temperature for specific heat \sqrt{C_V},
      derive the ladder density eta as a discrete function of temperature.
    Inputs:
        num_chains: Number of chains in the temperature ladder.
        scaling: An overall scaling factor for eta with respect to \sqrt{C_V}.
        Cv_func: Akima1DInterpolator object of the derivative of -avg_lnlikes with respect to temp_ladder.
    Steps:
        1. Cv_func (an Akima1DInterpolator object) is assumed to represent the ladder density, eta=1/\log_{10}\gamma
        2. In accordance with the above step, iteratively construct a list of etas based on sqrtCv(T), starting with eta(T=1)
        3. Enforce normalization condition that \int_1^{\log T_{max}}\eta d(\log T) = N_c = (num_chains)'''
    # Initialize etas, gammas
    etas=np.zeros(num_chains) # NOTE eta(T_max) is undefined
    gammas=np.full(num_chains,1.0)

    # Initialize first eta+gamma
    etas[0] = np.sqrt(Cv_func(1)) # Cv_func(1) means $C_V(T=1)$
    gammas[0] = 10**(1./etas[0]) # 10^{1/\eta}=\gamma,

    # Now iteratively construct the remaining gammas
    for i in range(1,num_chains):
        next_temp = np.prod(gammas) # \gamma_i = T_{i+1}/T_i, so T_{i} = T_{i-1}\gamma_{i-1} = T_{i-2}\gamma_{i-2}\gamma_{i-1} = T_0\gamma_0...\gamma_{i-1} = \prod_{k=0}^{i-1} \gamma_k
        # print(f'T_{i} = {next_temp}, C_V(T_{i}) = {Cv_func(next_temp)}')

        etas[i] = max(1.1, scaling*np.sqrt(Cv_func(next_temp))) # assume \eta = \sqrt{C_V}
        # print(f'eta_{i} ~ {scaling}*\sqrt(C_V) = {etas[i]}')

        gammas[i] = min(8.11, 10.**(1./etas[i]))
        # print(f'gamma_{i} = {gammas[i]}','\n')

    gammas[-1]=np.inf # fix last gamma so it isn't 1 anymore
    etas[-1] = 0

    # Sanity check
    # print(f'gammas = {gammas}')

    # Normalize by appending an approximate factor of num_chains / ( \int_0^{\log_{10}T_{max}}\eta d(\log_{10}T)) to etas
    # Sanity check
    # print(f'etas (non-norm) = {etas}')


    temps = [1] # need temps to integrate with respect to
    for gamma in gammas[:-1]:
        temps.append(temps[-1]*gamma)    
    # print(f'temps = {temps}, shape {len(temps)}')
    logtemps = np.log10(temps)

    integral = np.trapezoid(etas, logtemps, dx=0.1)
    etas = (num_chains/integral) * etas

    # Sanity check
    # print(f'integral={integral}, Nc={num_chains}')
    # Sanity check
    # print(f'etas (norm) = {etas}')

    return temps, etas

        

#******************************************************** ********************************************************
#*********************** Evidence *********************** ********************************************************
#******************************************************** ********************************************************

class TIData:

    def __init__(self, betas, lnlikes, avg_lnlikes, lnlikes_stdevs, _Alpha5Ln,_Alpha6Ln, color, filenum):
        self.betas=betas
        self.lnlikes=lnlikes # want this to be a 2D array where each column denotes a beta value with its list of num_samples lnlikes
        self.avg_lnlikes=avg_lnlikes
        self.lnlikes_stdevs=lnlikes_stdevs
        self._Alpha5Ln=_Alpha5Ln
        self._Alpha6Ln=_Alpha6Ln
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
    return np.transpose(lnlikes) # <- what's up with this?

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
    def __init__(self, samples, _Alpha5Ln,_Alpha6Ln):
        # Basic attributes
        self.samples=samples
        self._Alpha5Ln=_Alpha5Ln
        self._Alpha6Ln=_Alpha6Ln

        # Dependent quantities
        self.Ms=self.get_Ms(self.samples)                     # shape: (num_chains, num_samples)
        self.fIMs=imrc.PHI_fJoin_INS/(self.Ms*imrc.MTSUN_SI)  # shape: (num_chains, num_samples)
        self.thSNR=self.get_thSNR()                          # shape: (num_chains, num_samples)

        # Fast thSNRs getter (jit)?

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
        return self.Ms, self.fIMs, self.thSNR

    def get_thSNR(self):
        '''Returns theoretical SNR (h|h), where h=h(x_inj).'''
        h = wg.FD_waveform(d.x_inj, self._Alpha5Ln, self._Alpha6Ln) # theoretical SNR need only be computed for the exact injected waveform
        snr = np.sqrt(l.fast_inner(h, h))

        # Sanity check tempered SNRs? ########
        # NOTE tempered SNR routine is here jic I need it
        # for T in np.geomspace(1,1.e6,20):
        #     print(f'SNR/np.sqrt({T}) = {snr/np.sqrt(T)}')
        ##################################################

        return snr

# vv Unnecessary for evidence calculations
def phasediff_vs_freq(alphas1, alphas2, ax:plt.Axes):
    '''Calculate and plot (as a fxn of frequency) the phase difference \Psi_2-\Psi1 between two waveforms h22_2 and h22_1, 
    with the same injected parameters but differing degrees of suppression. Suppression for each waveform is controlled 
    through alphas1 and alphas2, each a list of Alpha5Ln, Alpha6Ln.'''
    h22_1 = wg.get_h22(d.x_inj, alphas1[0],alphas1[1])
    h22_2 = wg.get_h22(d.x_inj, alphas2[0],alphas2[1])
    delta_psi = h22_2.phase - h22_1.phase
    
    ax.plot(wg.f,delta_psi, color='r')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel(r'$\Delta\Psi(f)=\Psi_{2}(f)-\Psi_{1}(f)$')
    ax.set_title(rf'$\alpha_1$={alphas1}, $\alpha_2$={alphas2}') # assume vec notation ig

def chain_heatmap(temp_ladder, lnposts, num_chains, ax:plt.Axes):
    '''Plots likelihood values of chains (without temperature scaling).
        temp_ladder: An output of PTMCMC().
        lnposts: An output of PTMCMC().
        num_chains: An input to PTMCMC().'''
    chain_colors = list(reversed([plt.cm.plasma(i / num_chains) for i in range(1, num_chains + 1)]))

    for j, (temp, color) in enumerate(zip(temp_ladder[::-1], chain_colors)):
        ax.plot(lnposts[::-1][j] * temp, color=color, label=f'T = {round(temp, 3)}')
    ax.set_xlabel('MCMC iteration')
    ax.set_ylabel(r'$log\mathcal{L}$')
    ax.legend(loc='lower right')

def trace_temp1chain(samples, _Alpha5Ln,_Alpha6Ln, ax:plt.Axes):
    '''Creates a trace plot for T = 1 chain.
        samples: An output of PTMCMC().'''
    linestyles={0: 'solid', 1: 'dashed'}
    for i in range(wg.ndim): # NOTE the coloring routine in this loop used just Lambda before... if the output gets hairy when _Alpha6Ln is left alone in the analysis then fix here
        ax.plot(samples[0,:,i], color=f'C{i+int(wg.ndim * _Alpha6Ln / 2)}', alpha=0.3)
        ax.axhline(d.x_inj[i],  color=f'C{i+int(wg.ndim * _Alpha6Ln / 2)}', alpha=0.9, label=wg.x_labels[i]+rf', $\alpha_5ln=${_Alpha5Ln}, $\alpha_6ln=${_Alpha6Ln}', linestyle=linestyles[_Alpha6Ln])
    ax.set_xlabel('MCMC iteration', fontsize=14)
    ax.set_ylabel('parameter value', fontsize=14)
    ax.legend(loc='lower right')

def cornerplots(samples, _Alpha5Ln=1,_Alpha6Ln=1):
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
    Fisher = j.Fisher(d.x_inj, _Alpha5Ln,_Alpha6Ln)
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

def avglnlikes_vs_temps(temp_ladder, lnposts, num_samples):
    '''returns average log likelihoods vs temperature, for each chain, for ONE set of suppression parameters.'''
    burnin = num_samples // 5
    avg_lnlikes = np.mean(lnposts[:, burnin:], axis=1) * temp_ladder # these hold 80% of all num_samples lnlikes
    lnlikes_stdevs = np.std(lnposts[:, burnin:], axis=1) * temp_ladder # ^^

    return temp_ladder, avg_lnlikes, lnlikes_stdevs


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