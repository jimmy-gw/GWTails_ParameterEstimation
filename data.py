'''Script define injected parameters, data, and associated objects.'''


import numpy as np
import wave_gen as wg


# injected parameters
m1_inj = 60.
m2_inj = 34.
m_inj = m1_inj+m2_inj
sigma_m_inj=1.915141153050287 # approximate uncertainty on total mass
tIM_inj=0.018/m_inj
chi1_inj = -0.11 # <- S1_inj = chi1*m1**2 = ?
chi2_inj = 0.63 # <- S2_inj = chi2*m2**2 = ?
log_dist_inj = 59.333 # 58.07
phic_inj = np.pi / 1.2
tc_inj = 0.5
x_inj = np.array([m1_inj, m2_inj, chi1_inj, chi2_inj, log_dist_inj, phic_inj, tc_inj])
#x_inj_inclDQs = np.array([m1_inj, m2_inj, chi1_inj, chi2_inj, log_dist_inj, phic_inj, tc_inj, m_inj, tIM_inj])

# make data waveform objects
data_h22 = wg.get_h22(x_inj)
data_h22_suppressed = wg.get_h22(x_inj,_Lambda=1)
    
# amplitude and phase of data in frequency-domain
data_amp, data_phase = np.array(data_h22.amp), np.array(data_h22.phase)
data_amp_suppressed, data_phase_suppressed = np.array(data_h22_suppressed.amp), np.array(data_h22_suppressed.phase)

data_FD_waveform = wg.FD_waveform(x_inj)
data_FD_waveform_suppressed = wg.FD_waveform(x_inj,_Lambda=1)
