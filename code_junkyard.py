# from TAILSvsNOTAILS_MCMC.ipynb
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Splining <lnL> vs \beta for evidence calculation

################## SCIPY IS BS; USE thermo_error.py #####################
# thermo_error.py only takes in a series of chains if they're organized in a .dat file in the format of logLchain.dat.
# It has helper functions for writing numpy arrays as .txt files
# To run thermo_error I need to import argparse, pathlib, and __future__ (?)


fig, ax = plt.subplots(figsize=(14,7))
num_mciters=50
for i in range(num_lmcmciters):
    # make sure betas_i, avg_lnlikes_i, and lnlikes_stdevs_i are initialized and stored in previous cell
    TIData = __TIData__[i]
    betas, avg_lnlikes, lnlikes_stdevs = TIData.betas, TIData.avg_lnlikes, TIData.lnlikes_stdevs
    TIData.evidence=np.zeros(num_mciters)

    # generate randomly-perturbed <lnL> v \beta plots
    for j in range(num_mciters):
        avg_lnlikes_akima_in = TIData.generatePerturbedLogLikesToSpline()
        betas_akima = np.linspace(min(betas), max(betas), num=100000)
        # Akima-spline the perturbed curve
        avg_lnlikes_akima_out = Akima1DInterpolator(betas, avg_lnlikes_akima_in, method="akima")
        # integrate under curve
        TIData.evidence[j]=avg_lnlikes_akima_out.integrate(min(betas),max(betas), extrapolate=False)
        # plot Akima-splined <lnL> v \beta
        ax.plot(betas_akima, avg_lnlikes_akima_out(betas_akima), marker=None, color=TIData.color, alpha=0.1)
    
    #print(f'Log-Evidence samples from lambdavec iteration #{i+1}: {TIData.evidence}')


#########################################################
fig.legend(loc='lower right', bbox_to_anchor=(0.98,0.02))
# ^^ make custom labels ^^ ##############################

ax.set_title(r'MC Splining of $\langle\log\mathcal{L}\rangle$ vs. $\beta$', size=30)
ax.set_xlabel(r'$\beta = 1/T$', size=20)    
ax.set_ylabel(r'$\langle\log\mathcal{L}\rangle$', size=20)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(alpha=0.3)

ax.set_xlim(-0.0002,0.02)

plt.show()

fig2,ax2=plt.subplots(figsize=(10,7))
for TIData in __TIData__:
    ax2.hist(TIData.evidence, color=TIData.color, alpha=0.5)
ax2.set_title(r'MC Measurements of $\log\mathcal{Z}$', size=20)
ax2.set_xlabel(r'$\log\mathcal{Z}$', size=20)    
ax2.set_ylabel(r'$N$', size=20)
ax2.tick_params(axis='both', which='major', labelsize=14)
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


### MANUAL ladder spacing optimization (NOT adaptive)
# Perform PTMCMCs and calculate mean PT swap rates w/ variance for S random num_chain and T0 values, then make a SCATTERPLOT with COLORS

S=50 # computation time is about 2hr/100 iterations

# independent variables
Ncs=np.zeros(S)
T0s=np.zeros(S)
# dependent variables
swaprate_avgs=np.zeros(S)
swaprate_stds=np.zeros(S)

fig, (ax1, ax2)=plt.subplots(1,2, figsize=(14,6))

### Get PT swap acceptance rates
for i in range(S):
    print(rf'######### Running PTMCMC #{i+1}/{S} #########')
    ## Do PTMCMC without any suppression for simplicity
    num_chains=np.random.randint(20,30)
    temp0=np.random.uniform(1.9,2.25)
    samples, lnposts, acc_rates = lmcmc.PTMCMC_i(lambdas=[0,0,0,0], 
                                                   temp_ladder=temp_ladder
                                                   num_samples=10000,
                                                   num_chains=num_chains,
                                                   temp0=temp0)

    ## save PT swap acceptance rates
    PTswapaccrate_avg=np.mean(acc_rates['PT_swap'][:-1])
    PTswapaccrate_var=np.std(acc_rates['PT_swap'][:-1])
    
    Ncs[i]=num_chains
    T0s[i]=temp0
    swaprate_avgs[i]=PTswapaccrate_avg
    swaprate_stds[i]=PTswapaccrate_var

### create plot for averages
sc1 = ax1.scatter(
    Ncs,
    T0s,
    c=swaprate_avgs,      # values that determine color
    cmap="magma",         # seaborn colormap
    s=25                  # marker size (optional)
)

cbar1 = plt.colorbar(sc1, ax=ax1)
cbar1.set_label("Mean PT Swap Acceptance Rate")

ax1.set_xlabel(r"Number of Chains $N_c$")
ax1.set_ylabel(r"$T_0$")

# create plot for variances
sc2 = ax2.scatter(
    Ncs,
    T0s,
    c=swaprate_stds,      # values that determine color
    cmap="viridis",         # seaborn colormap
    s=25                  # marker size (optional)
)

cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label("PT Swap Acceptance Rate Standard Deviation")

ax2.set_xlabel(r"Number of Chains $N_c$")
ax2.set_ylabel(r"$T_0$")
fig.suptitle("PT Swap Rate Across PTMCMC Configurations", fontsize=20)

plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# from lambdasMCMC.py
########################## attempt #1
def DEPRECATEDconstruct_temp_ladder(lambdas, temp_ladder, Tmax=1000000., counter=1):
    '''Iterative calculation of Eq. (15) of Rathore et al. 2004 [@DOI: 10.1063/1.1831273], which returns an optimal temperature ladder for PT swap acceptance rates in the range of 10-20%.'''    

    ### Initialize
    print(f'********************* RUNNING TEMP LADDER CONSTRUCTION ITERATION #{counter} *********************')
    num_samples=10000 # figure out what to do about this being 10K instead of 100K
    temp0=temp_ladder[1] # base temperature
    Rbar_tgt=1. # desired R-statistic indicative of 10-20% acceptance rate


    if temp_ladder[0]!=1:
        raise ValueError(f'temp_ladder[0]={temp_ladder[0]} must be equal to 1.')
    if temp0<=1:
        raise ValueError(f'temp_ladder[1]=temp0={temp0} must be greater than 1.')

    ### Run trial PTMCMC
    ## to get the correct next temperature in the ladder, I need the swap acceptance rates to be between 10 and 20 percent consistently.
    __, lnposts, acc_rates = PTMCMC_i(lambdas=lambdas, 
                                      temp_ladder=temp_ladder,
                                      num_samples=10000,
                                      num_chains=len(temp_ladder),
                                      return_acceptance_rates=True)
   
    ## Get relevant data from PTMCMC
    betas, avg_lnlikes, lnlikes_stdevs = avglnlikes_vs_betas(temp_ladder, lnposts, num_samples)

    # overlap factors
    alphas=np.zeros(len(temp_ladder)-1)
   for j in range(len(temp_ladder)-1):
      avg_lnlike_1, avg_lnlike_2 = avg_lnlikes[j], avg_lnlikes[j+1]
     lnlikes_stdev_1, lnlikes_stdev_2 = lnlikes_stdevs[j], lnlikes_stdevs[j+1]
    alphas[j]=alpha_overlap(avg_lnlike_1, lnlikes_stdev_1, avg_lnlike_2, lnlikes_stdev_2)

    # R-statistics normalized by alpha and not normalized by alpha
    #__,Rs=ladder_spacing_ratio(temp_ladder, avg_lnlikes, lnlikes_stdevs, alphas, print_chains=False, haxis='B')
    __,Rbars=ladder_spacing_ratio(temp_ladder, avg_lnlikes, lnlikes_stdevs, alphas=np.full(len(alphas),1.), print_chains=False, haxis='B')

    # PT chain swap acceptance rates
    Pacc=acc_rates['PT_swap']
    avgPTswapaccrate=np.mean(Pacc[:-1]) # does NOT include acceptance rate of swaps for the hottest chain, because that doesn't get swapped with anything hotter (duh!)
    #print(f'Average PT swap acceptance rate: {avgPTswapaccrate}')
    #print(f'~~~(temp_ladder, R)={list(zip(temp_ladder,R))}~~~')
    #print(f'~~~(temp_ladder, Pacc)={list(zip(temp_ladder,Pacc))}~~~')


    ### Iteratively adjust ladder spacing (in BETA space) until the temp ladder is complete
    for i in range(len(temp_ladder[:-1])):
        p=Pacc[i]
        if 0.1<=p<=0.2:
            continue
        else:
            # calculate new beta spacing based on the R statistic
            old_beta_spacing=betas[i+i]-betas[i]
            new_beta_spacing=old_beta_spacing*np.sqrt(Rbar_tgt/Rbars[i])
            shift=new_beta_spacing-old_beta_spacing
            # calculate what the temp ladder looks like with betas shifted
#   NO!! DONT WANT ENGATIVE BETAS THAT'S bad            betas=np.concatenate((betas[:i+1], betas[i+1:]+np.full(len(betas[i+1:], shift))))
            next_temp_ladder=1./betas
            construct_temp_ladder(lambdas, next_temp_ladder, Tmax=Tmax, counter=counter+1)


    # Make sure the hottest chain is hot enough (\beta_{min}~10^-6)
    if temp_ladder[-1]>=Tmax:
        ## Sanity checks
        print(f'****************##### Temperature ladder constructed successfully in {counter} iterations #####****************')
        print('Results:')
        print(f'\ttemp_ladder: {temp_ladder}')
        print(f'\tPacc: {Pacc}, avgPTswapaccrate: {avgPTswapaccrate}')
        print(f'\tR: {Rs}, Rbar: {Rbars}')
        return temp_ladder
    else:
        # if the hottest chain is too cold then just rerun the algo with another one added to the end in a geometric fashion
        print(f'***** New chain has been added at T={temp_ladder[-1]*temp0}*****')
        construct_temp_ladder(lambdas, temp_ladder.append(temp_ladder[-1]*temp0), Tmax=Tmax, counter=counter+1)