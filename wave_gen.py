'''Script to generate gravitational waveforms using IMRPhenomD.

IMPORTANT: the input parameters and those sampled with MCMC are (in order)
[mass 1, mass 2, spin 1, spin 2, log(luminosity distance), phase at coalesence].
Masses are measured in units of solar mass and luminosity distance is in meters.
All other parameters are dimensionless.'''


import numpy as np
import IMRPhenomD.IMRPhenomD_const as imrc
from IMRPhenomD.IMRPhenomD import AmpPhaseFDWaveform, IMRPhenomDGenerateh22FDAmpPhase
# import data as d


# parameter attributes
x_mins = np.array([2.5, 2.5, -0.97, -0.97, np.log(1.e8 * imrc.PC_SI), -3*np.pi, -30])
x_maxs = np.array([100.0, 100.0, 0.97, 0.97, np.log(1.e10 * imrc.PC_SI), 3*np.pi, 30])
ndim = x_mins.shape[0]
x_labels = [r'$m_1\;(M_\odot)$', r'$m_2\;(M_\odot)$', r'$\chi_1$', r'$\chi_2$', r'$\ln(D_L/1m)$', r'$\phi_c$', r'$t_c$']

x_labels_inclDQs = [r'$m_1\;(M_\odot)$', r'$m_2\;(M_\odot)$', r'$\chi_1$', r'$\chi_2$', r'$\ln(D_L/1m)$', r'$\phi_c$', r'$t_c$', r'$m$', r'$t_{IM}$'] # need SNR too (I think)


# frequency bins
f_min = 10.
f_max = 1024. # <- set to 0.018/(m+4\sigma_{mm}) 
# f_max = 0.018/(imrc.MTSUN_SI*(d.m_inj+4*d.sigma_m_inj)) (8/5/26)
# Nf = 2**9 + 1
Nf = 2**12 + 1
f = np.linspace(f_min, f_max, Nf)
df = f[1] - f[0]


# get waveform object (for l=2, m=2 mode)
# input parameters as array in order described above
def get_h22(x, _Alpha5Ln=1,_Alpha6Ln=1):
            
    # mass in solar masses
    # distance in ln(luminosity distance / meter)
    m1, m2, chi1, chi2, log_dist, phic, tc = x
    
    # convert masses in kg
    m1_SI =  m1*imrc.MSUN_SI
    m2_SI =  m2*imrc.MSUN_SI
    
    # distance in meters
    distance = np.exp(log_dist)

    # initialize amplitudes and times
    amp_imr = np.zeros(Nf)
    phase_imr = np.zeros(Nf)
    time_imr = np.zeros(Nf)
    timep_imr = np.zeros(Nf)

    # use peak reference frequency
    MfRef_in = 0

    #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
    h22 = AmpPhaseFDWaveform(Nf, f, amp_imr, phase_imr, time_imr, timep_imr)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22, f, phic, MfRef_in, m1_SI, m2_SI, chi1, chi2, distance, _Alpha5Ln,_Alpha6Ln)

    # Modulate phase by coalescence time shift
    h22.phase+=2.*np.pi*h22.freq*(tc/1000) # coalescence time tc is in units of ms

    return h22


# get frequency-domain waveform
def FD_waveform(x,_Alpha5Ln=1,_Alpha6Ln=1):
    h22 = get_h22(x, _Alpha5Ln,_Alpha6Ln)
    return h22.amp * np.exp(-1.j * h22.phase)


# compute partial derivative of frequency-domain waveform
# (used for Fisher evaluation)
def partial_FD_waveform(x, deriv_ndx, _Alpha5Ln=1,_Alpha6Ln=1, epsilon=3.e-8):
    # central finite differencing
    delta_x = np.zeros(ndim)
    delta_x[deriv_ndx] = epsilon*x[deriv_ndx]

    # sanity check that differential perturbations are consistent
    #print(f'x={x[deriv_ndx]}, delta_x={delta_x[deriv_ndx]}, ratio={delta_x[deriv_ndx]/x[deriv_ndx]}')

    deriv=(FD_waveform(x + delta_x, _Alpha5Ln,_Alpha6Ln) - FD_waveform(x - delta_x, _Alpha5Ln,_Alpha6Ln)) / (2. * delta_x[deriv_ndx])

    return deriv


