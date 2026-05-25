'''Script for generating only the logarithmic terms for gravitational 
waveforms using IMRPhenomD, up to suppression/toggling factors lambda25/
lambda3, to be subtracted from the waveforms utilized in example.ipynb.'''

import numpy as np
from numba import njit
from IMRPhenomD.IMRPhenomD_fring_helper import fring_interp,fdamp_interp,EradRational0815
import IMRPhenomD.IMRPhenomD_const as imrc

def PNPhasingLogTermsTaylorF2(eta,chis,chia, lambda25=0, lambda3=0):
    """Mimics the function PNPhasingSeriesTaylorF2 but returns only that 
    fraction of the log terms at 2.5 and 3PN orders to be suppressed. The 
    toggling parameters lambda25 and lambda3 denote the fraction of the 
    log terms that is to be suppressed. 0->recover GR, 1->remove logs entirely"""

    if eta<0.25:
        delta = np.sqrt(1-4*eta)
    else:
        delta = 0.

    #Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
    #We write dSigmaL for their (\delta m/m) * \Sigma_\ell
    #There's a division by mtotal^2 in both the energy and flux terms
    #We just absorb the division by mtotal^2 into SL and dSigmaL/

    SL = chis*(1-2*eta) + chia*delta
    dSigmaL = -delta*(chis*delta + chia)

    pfaN = 3/(128*eta)

#    /* Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
    # # # # # Logarithmic terms at 2.5PN, 3PN order - see arXiv:0907.0700, Eq. 3.18 # # # # #
    # 2.5PN order corrections to 1st-order/parent tail
    vlogv5 = 5/3*np.pi*(7729/84-13*eta)
    # 3PN order tail-of-tail contribution
    vlogv6 = -6848/21 
    
#     Compute 2.0PN SS, QM, and self-spin */
#     See Eq. (6.24) in arXiv:0810.5336
#     9b,c,d in arXiv:astro-ph/0504538
    pn_sigma = 1/16*(chia**2*(81 - 320*eta) + chis**2*(81 - 4*eta) + 162*chis*chia*delta)

    if imrc.include3PNSS:
        pn_ss3 = 1/2016*(70*chia*chis*delta*(15103 - 13160*eta) \
                        + 5*chis**2*(105721 + 4*eta*(-46483 + 14056*eta)) \
                        - 5*chia**2*(-105721 + 8*eta*(52649 + 24192*eta)))
    else:
        pn_ss3 = 0.

#     Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
    pn_gamma = (554345/1134+110/9*eta)*SL+(13915/84-10/3*eta)*dSigmaL

    vlogv5 += -3*pn_gamma

#     At the very end, multiply everything in the series by pfaN */
    v = (0.,0.,0.,0.,0.,0.,0.,0.)
    vlogv = (0.,0.,0.,0.,0.,lambda25*pfaN*vlogv5,lambda3*pfaN*vlogv6,0.)

    return v,vlogv

# confer notes on where to take v,logv object from here to make suppresion terms in the end