"""
    Codes to preform numerical integration of precession ODEs. 
    Copyright (C) 2023,  Hang Yu (hang.yu2@montana.edu)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import scipy.special as special
import h5py as h5
from numba import jit, prange, njit

from .LAL_constants import *


########

@njit(fastmath=True, cache=True)
def inner(xx, yy):
    return np.sum(xx*yy)

@njit(fastmath=True, cache=True)
def cross(xx, yy):
    zz=np.array([\
         xx[1]*yy[2] - xx[2]*yy[1], \
         xx[2]*yy[0] - xx[0]*yy[2], \
         xx[0]*yy[1] - xx[1]*yy[0]
                ])
    return zz

@njit
def rot_z(al, vv_in):
    ca = np.cos(al)
    sa = np.sin(al)
    
    vv_out = np.array([
        ca * vv_in[0] - sa * vv_in[1], 
        sa * vv_in[0] + ca * vv_in[1], 
        vv_in[2]
    ])
    return vv_out


########


@njit(fastmath=True, cache=True)
def get_dMw_dt_M(M_omega_pows, aa, bb):
    """
    GW decay (no e precession as the orb is circ)
    from Chatziioannou+ 13, PRD 88, 063011
    """    
    # scalars
    M_omega_2_3 = M_omega_pows[1]
    M_omega_3_3 = M_omega_pows[2]
    M_omega_4_3 = M_omega_pows[3]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_6_3 = M_omega_pows[5]
    M_omega_7_3 = M_omega_pows[6]
    M_omega_8_3 = M_omega_pows[7]
    
    log_M_omega = np.log(M_omega_3_3)
    
    # adding a9 and b9 seems to make things worse
    a0, a2, a3, a4, a5, a6, a7, a8 \
        = aa[0], aa[1], aa[2], aa[3], aa[4], aa[5], aa[6], aa[7]
    b6, b8 \
        = bb[0], bb[1]    
    
    dMw_dt_M = M_omega_6_3 * a0 * M_omega_5_3\
            * (1. \
               + a2 * M_omega_2_3 + a3 * M_omega_3_3 + a4 * M_omega_4_3\
               + a5 * M_omega_5_3 + (a6 + b6 * log_M_omega) * M_omega_6_3\
               + a7 * M_omega_7_3\
               + (a8 + b8 * log_M_omega) * M_omega_8_3)

    return dMw_dt_M


@njit(fastmath=True, cache=True)
def get_inv_dMw_dt_M(M_omega_pows, gg, gl):
    """
    1/dMw_dt_M
    directly computing from 1/get_dMw_dt_M could have numerical issues when dMw_dt_M = 0
    
    from Chatziioannou+ 17, PRD 95, 104004
    """
    
    # scalars
    M_omega_2_3 = M_omega_pows[1]
    M_omega_3_3 = M_omega_pows[2]
    M_omega_4_3 = M_omega_pows[3]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_6_3 = M_omega_pows[5]
    M_omega_7_3 = M_omega_pows[6]
    M_omega_8_3 = M_omega_pows[7]
    
    log_M_omega = np.log(M_omega_3_3)
    
    inv_M_omega_11_3 = 1./(M_omega_5_3 * M_omega_6_3)
    
    g0, g2, g3, g4, g5, g6, g7, g8\
        = gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], gg[6], gg[7]
    gl6, gl8 = gl[0], gl[1]
    
    
    
    inv_dMw_dt_M = inv_M_omega_11_3 \
            * (g0 \
               + g2 * M_omega_2_3 + g3 * M_omega_3_3\
               + g4 * M_omega_4_3 + g5 * M_omega_5_3\
               + (g6 + gl6 * log_M_omega) * M_omega_6_3\
               + g7 * M_omega_7_3\
               + (g8 + gl8 * log_M_omega) * M_omega_8_3)
    return inv_dMw_dt_M
    

@njit(fastmath=True, cache=True)
def get_dy_dMw_SP(y_SP_vect, M_omega_pows, par_circ, par_SP_circ):
    """
    spin precession
    """
    # parse input
    uL_v = y_SP_vect[0:3]
    uS1_v = y_SP_vect[3:6]
    uS2_v = y_SP_vect[6:9]
    
    # global par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
        
    # par for spin precession calc
    inv_L_N_Mt, inv_dMw_dt_M, \
    uL_d_uS1, uL_d_uS2, uS1_d_uS2\
            = par_SP_circ
    
    M_omega_1_3 = M_omega_pows[0]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_2   = M_omega_pows[5]
    
    # directional products 
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    d_uS1_v_dt_M = (eta * (2. + 1.5*qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 * qq + S2_Mt*uL_d_uS2 ) ) \
              * M_omega_5_3 * uL_c_uS1_v\
            - 0.5 * M_omega_2 * S2_Mt * uS1_c_uS2_v
    
    d_uS2_v_dt_M = (eta * (2. + 1.5/qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 + S2_Mt*uL_d_uS2 / qq ) ) \
              * M_omega_5_3 * uL_c_uS2_v\
            + 0.5 * M_omega_2 * S1_Mt * uS1_c_uS2_v
    
    d_uS1_v_dMw = d_uS1_v_dt_M * inv_dMw_dt_M
    d_uS2_v_dMw = d_uS2_v_dt_M * inv_dMw_dt_M
    
    d_uL_v_dMw = - S1_Mt * inv_L_N_Mt * d_uS1_v_dMw - S2_Mt * inv_L_N_Mt * d_uS2_v_dMw
    
    dy_dMw_SP = np.hstack((d_uL_v_dMw, d_uS1_v_dMw, d_uS2_v_dMw)) 
    return dy_dMw_SP

@njit(fastmath=True, cache=True)
def get_dy_dMw_SP_N4LO(y_SP_vect, M_omega_pows, par_circ, par_SP_circ):
    """
    spin precession
    + N4LO dyn from https://arxiv.org/pdf/2005.05338.pdf
    """
    # parse input
    uL_v = y_SP_vect[0:3]
    uS1_v = y_SP_vect[3:6]
    uS2_v = y_SP_vect[6:9]
    
    # global par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # par for spin precession calc
    inv_L_N_Mt, inv_dMw_dt_M, \
    uL_d_uS1, uL_d_uS2, uS1_d_uS2\
            = par_SP_circ
    
    # useful powers of M_omega
    M_omega_1_3 = M_omega_pows[0]
    M_omega_2_3 = M_omega_pows[1]
    M_omega     = M_omega_pows[2]
    M_omega_4_3 = M_omega_pows[3]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_2   = M_omega_pows[5]
    M_omega_7_3 = M_omega_pows[6]
    M_omega_3   = M_omega_pows[8]
    
    # directional products 
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)

    #
    delta = (M1-M2)/Mt
    eta2 = eta * eta
    eta3 = eta2 * eta
    
    S1_L = S1_Mt * inv_L_N_Mt
    S2_L = S2_Mt * inv_L_N_Mt
    
    cS1 = - 0.75 - 0.25*Mt/M1
    cS1L = - 0.08333333333 - 2.25*Mt/M1
    cS2 = - 0.75 - 0.25*Mt/M2
    cS2L = - 0.08333333333 - 2.25*Mt/M2
    
    inv_L_2PN = 1./(1. + M_omega_2_3 * (1.5 + 0.1666666666667*eta) \
                       + M_omega_4_3 * (3.375 - 2.375*eta + 0.04166666666667*eta2))
    
    # full evol in the inertial frame
    # NLO
    d_uS1_v_dt_M_NLO = (eta * (2. + 1.5*qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 * qq + S2_Mt*uL_d_uS2 ) ) \
              * M_omega_5_3 * uL_c_uS1_v\
            - 0.5 * M_omega_2 * S2_Mt * uS1_c_uS2_v
    
    d_uS2_v_dt_M_NLO = (eta * (2. + 1.5/qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 + S2_Mt*uL_d_uS2 / qq ) ) \
              * M_omega_5_3 * uL_c_uS2_v\
            + 0.5 * M_omega_2 * S1_Mt * uS1_c_uS2_v
    
    # NNLO
    d_uS1_v_dt_M_NNLO = d_uS1_v_dt_M_NLO \
            + M_omega_7_3 \
                * (0.5625 + 1.25*eta - 0.04166666667*eta2 + delta * (-0.5625 + 0.625*eta))\
                * uL_c_uS1_v
    d_uS2_v_dt_M_NNLO = d_uS2_v_dt_M_NLO \
            + M_omega_7_3 \
                * (0.5625 + 1.25*eta - 0.04166666667*eta2 - delta * (-0.5625 + 0.625*eta))\
                * uL_c_uS2_v
    
    # N4LO
    d_uS1_v_dt_M = d_uS1_v_dt_M_NNLO \
            + M_omega_3 \
                * ( 0.84375 + 0.1875*eta - 3.28125*eta2 - 0.02083333333333*eta3\
                  + delta*(-0.84375 + 4.875*eta - 0.15625*eta2))\
                * uL_c_uS1_v
    d_uS2_v_dt_M = d_uS2_v_dt_M_NNLO \
            + M_omega_3 \
                * ( 0.84375 + 0.1875*eta - 3.28125*eta2 - 0.02083333333333*eta3\
                  - delta*(-0.84375 + 4.875*eta - 0.15625*eta2))\
                * uL_c_uS2_v
    
    # back-reaction
    d_uL_v_dt_M_NLO = - S1_L * d_uS1_v_dt_M - S2_L * d_uS2_v_dt_M
    d_uL_v_dt_M = inv_L_2PN \
        * (-S1_L * (d_uS1_v_dt_M \
                    + eta * M_omega_2_3 \
                      * (cS1 * d_uS1_v_dt_M_NNLO + cS1L * uL_d_uS1 * d_uL_v_dt_M_NLO))\
           -S2_L * (d_uS2_v_dt_M \
                    + eta * M_omega_2_3 \
                      * (cS2 * d_uS2_v_dt_M_NNLO + cS2L * uL_d_uS2 * d_uL_v_dt_M_NLO))) 
    
    # to /dMw
    d_uS1_v_dMw = inv_dMw_dt_M * d_uS1_v_dt_M
    d_uS2_v_dMw = inv_dMw_dt_M * d_uS2_v_dt_M
    d_uL_v_dMw  = inv_dMw_dt_M * d_uL_v_dt_M
    
    dy_dMw_SP = np.hstack((d_uL_v_dMw, d_uS1_v_dMw, d_uS2_v_dMw))
    return dy_dMw_SP


@njit(fastmath=True, cache=True)
def get_dy_dMw_SP_fast(M_omega_pows, cross_prods, par_circ, par_SP_fast):
    """
    spin precession
    fast version
    """
    
    # global par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # useful powers of M_omega
    M_omega_1_3 = M_omega_pows[0]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_2   = M_omega_pows[5]
    
    # directional products 
    uL_c_uS1_v = cross_prods[0, :]
    uL_c_uS2_v = cross_prods[1, :]
    uS1_c_uS2_v = cross_prods[2, :]
    uJ0_c_uL_v = cross_prods[3, :]
    uJ0_c_uS1_v = cross_prods[4, :]
    uJ0_c_uS2_v = cross_prods[5, :]

    # additional inputs computed in the main evol func
    inv_L_N_Mt, inv_dMw_dt_M, \
    uL_d_uS1, uL_d_uS2, uS1_d_uS2, \
    al_0, dal_0_dMw\
            = par_SP_fast
        
    # full evol in the inertial frame
    d_uS1_v_dt_M = (eta * (2. + 1.5*qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 * qq + S2_Mt*uL_d_uS2 ) ) \
              * M_omega_5_3 * uL_c_uS1_v\
            - 0.5 * M_omega_2 * S2_Mt * uS1_c_uS2_v
    
    d_uS2_v_dt_M = (eta * (2. + 1.5/qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 + S2_Mt*uL_d_uS2 / qq ) ) \
              * M_omega_5_3 * uL_c_uS2_v\
            + 0.5 * M_omega_2 * S1_Mt * uS1_c_uS2_v
    
    d_uS1_v_dMw = d_uS1_v_dt_M * inv_dMw_dt_M
    d_uS2_v_dMw = d_uS2_v_dt_M * inv_dMw_dt_M
    
    d_uL_v_dMw = - S1_Mt * inv_L_N_Mt * d_uS1_v_dMw - S2_Mt * inv_L_N_Mt * d_uS2_v_dMw
    
    # subtracting the alpha rotation
    d_uS1_v_dMw -= dal_0_dMw * uJ0_c_uS1_v
    d_uS2_v_dMw -= dal_0_dMw * uJ0_c_uS2_v
    d_uL_v_dMw -= dal_0_dMw * uJ0_c_uL_v
    
    # do not need to go to the co-rot frame,
    # if the cross prods are already in the co-rot frame!
    
#     d_uS1_r_v_dMw = rot_z(-al_0, d_uS1_v_dMw)
#     d_uS2_r_v_dMw = rot_z(-al_0, d_uS2_v_dMw)
#     d_uL_r_v_dMw = rot_z(-al_0, d_uL_v_dMw)
    
    # return the dyn in the precessing frame
    dy_dMw_SP = np.hstack((d_uL_v_dMw, d_uS1_v_dMw, d_uS2_v_dMw))
    return dy_dMw_SP


@njit(fastmath=True, cache=True)
def get_dy_dMw_SP_fast_N4LO(M_omega_pows, cross_prods, par_circ, par_SP_fast):
    """
    spin precession
    fast version 
    + N4LO dyn from https://arxiv.org/pdf/2005.05338.pdf
    """
    
    # global par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # useful powers of M_omega
    M_omega_1_3 = M_omega_pows[0]
    M_omega_2_3 = M_omega_pows[1]
    M_omega     = M_omega_pows[2]
    M_omega_4_3 = M_omega_pows[3]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_2   = M_omega_pows[5]
    M_omega_7_3 = M_omega_pows[6]
    M_omega_3   = M_omega_pows[8]
    
    # directional products 
    uL_c_uS1_v = cross_prods[0]
    uL_c_uS2_v = cross_prods[1]
    uS1_c_uS2_v = cross_prods[2]
    uJ0_c_uL_v = cross_prods[3]
    uJ0_c_uS1_v = cross_prods[4]
    uJ0_c_uS2_v = cross_prods[5]

    # additional inputs computed in the main evol func
    inv_L_N_Mt, inv_dMw_dt_M, \
    uL_d_uS1, uL_d_uS2, uS1_d_uS2, \
    al_0, dal_0_dMw\
            = par_SP_fast
    
    #
    delta = (M1-M2)/Mt
    eta2 = eta * eta
    eta3 = eta2 * eta
    
    S1_L = S1_Mt * inv_L_N_Mt
    S2_L = S2_Mt * inv_L_N_Mt
    
    cS1 = - 0.75 - 0.25*Mt/M1
    cS1L = - 0.08333333333 - 2.25*Mt/M1
    cS2 = - 0.75 - 0.25*Mt/M2
    cS2L = - 0.08333333333 - 2.25*Mt/M2
    
    inv_L_2PN = 1./(1. + M_omega_2_3 * (1.5 + 0.1666666666667*eta) \
                       + M_omega_4_3 * (3.375 - 2.375*eta + 0.04166666666667*eta2))
    
    # full evol in the inertial frame
    # NLO
    d_uS1_v_dt_M_NLO = (eta * (2. + 1.5*qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 * qq + S2_Mt*uL_d_uS2 ) ) \
              * M_omega_5_3 * uL_c_uS1_v\
            - 0.5 * M_omega_2 * S2_Mt * uS1_c_uS2_v
    
    d_uS2_v_dt_M_NLO = (eta * (2. + 1.5/qq) \
               - 1.5 * M_omega_1_3 * ( S1_Mt*uL_d_uS1 + S2_Mt*uL_d_uS2 / qq ) ) \
              * M_omega_5_3 * uL_c_uS2_v\
            + 0.5 * M_omega_2 * S1_Mt * uS1_c_uS2_v
    
    # NNLO
    d_uS1_v_dt_M_NNLO = d_uS1_v_dt_M_NLO \
            + M_omega_7_3 \
                * (0.5625 + 1.25*eta - 0.04166666667*eta2 + delta * (-0.5625 + 0.625*eta))\
                * uL_c_uS1_v
    d_uS2_v_dt_M_NNLO = d_uS2_v_dt_M_NLO \
            + M_omega_7_3 \
                * (0.5625 + 1.25*eta - 0.04166666667*eta2 - delta * (-0.5625 + 0.625*eta))\
                * uL_c_uS2_v
    
    # N4LO
    d_uS1_v_dt_M = d_uS1_v_dt_M_NNLO \
            + M_omega_3 \
                * ( 0.84375 + 0.1875*eta - 3.28125*eta2 - 0.02083333333333*eta3\
                  + delta*(-0.84375 + 4.875*eta - 0.15625*eta2))\
                * uL_c_uS1_v
    d_uS2_v_dt_M = d_uS2_v_dt_M_NNLO \
            + M_omega_3 \
                * ( 0.84375 + 0.1875*eta - 3.28125*eta2 - 0.02083333333333*eta3\
                  - delta*(-0.84375 + 4.875*eta - 0.15625*eta2))\
                * uL_c_uS2_v
    
    # back-reaction
    d_uL_v_dt_M_NLO = - S1_L * d_uS1_v_dt_M - S2_L * d_uS2_v_dt_M
    d_uL_v_dt_M = inv_L_2PN \
        * (-S1_L * (d_uS1_v_dt_M \
                    + eta * M_omega_2_3 \
                      * (cS1 * d_uS1_v_dt_M_NNLO + cS1L * uL_d_uS1 * d_uL_v_dt_M_NLO))\
           -S2_L * (d_uS2_v_dt_M \
                    + eta * M_omega_2_3 \
                      * (cS2 * d_uS2_v_dt_M_NNLO + cS2L * uL_d_uS2 * d_uL_v_dt_M_NLO))) 
    
    # to /dMw
    d_uS1_v_dMw = inv_dMw_dt_M * d_uS1_v_dt_M
    d_uS2_v_dMw = inv_dMw_dt_M * d_uS2_v_dt_M
    d_uL_v_dMw  = inv_dMw_dt_M * d_uL_v_dt_M
    
    # subtracting the alpha rotation
    d_uS1_v_dMw -= dal_0_dMw * uJ0_c_uS1_v
    d_uS2_v_dMw -= dal_0_dMw * uJ0_c_uS2_v
    d_uL_v_dMw -= dal_0_dMw * uJ0_c_uL_v
    
    # do not need to go to the co-rot frame,
    # if the cross prods are already in the co-rot frame!
    
#     d_uS1_r_v_dMw = rot_z(-al_0, d_uS1_v_dMw)
#     d_uS2_r_v_dMw = rot_z(-al_0, d_uS2_v_dMw)
#     d_uL_r_v_dMw = rot_z(-al_0, d_uL_v_dMw)

    # return the dyn in the precessing frame
    dy_dMw_SP = np.hstack((d_uL_v_dMw, d_uS1_v_dMw, d_uS2_v_dMw))
    return dy_dMw_SP


@njit(fastmath=True, cache=True)
def get_al_dal_0(M_omega_pows, par_simp_prec,
                 al_offset=0):
    
    A_n3, A_n2, A_n1 \
            = par_simp_prec
    
    #
    M_omega_1_3 = M_omega_pows[0]
    M_omega_2_3 = M_omega_pows[1]
    M_omega_3_3 = M_omega_pows[2]
    M_omega_4_3 = M_omega_pows[3]
    M_omega_5_3 = M_omega_pows[4]
    M_omega_6_3 = M_omega_pows[5]
    
    
    # get the leading order azimuth angle
    al_0 = A_n3 / M_omega_3_3 + A_n2 / M_omega_2_3 + A_n1 / M_omega_1_3
    al_0 += al_offset
    
    dal_0_dMomega = - A_n3 / M_omega_6_3 \
        - 0.66666666667 * A_n2 / M_omega_5_3 - 0.33333333333 * A_n1 / M_omega_4_3
    return al_0, dal_0_dMomega


@njit(fastmath=True, cache=True)
def evol_binary_circ(M_omega, y_nat_vect, par_circ):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    # updating the PN coefficients as uL_d_uS1, etc, may have changed
    aa, bb = get_PN_coeff(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2)
#     gg, gl = get_gg_gl_from_aa_bb(aa, bb)
    
#     # get GW evolution of the frequency
#     inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)

    inv_dMw_dt_M = 1./get_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_circ = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2])
    dy_dMw = get_dy_dMw_SP(y_nat_vect, M_omega_pows, par_circ, par_SP_circ)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_N4LO(M_omega, y_nat_vect, par_circ):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    # updating the PN coefficients as uL_d_uS1, etc, may have changed
    aa, bb = get_PN_coeff(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2)
#     gg, gl = get_gg_gl_from_aa_bb(aa, bb)
    
#     # get GW evolution of the frequency
#     inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)

    inv_dMw_dt_M = 1./get_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_circ = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2])
    dy_dMw = get_dy_dMw_SP_N4LO(y_nat_vect, M_omega_pows, par_circ, par_SP_circ)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast(M_omega, y_nat_vect, par_circ, par_simp_prec, 
                          al_offset=0):
    """
    Fast precession evolution by solving dynamics in a frame precessing around J_0_v = [0, 0, 1]
    with a phase given by al_0
    """
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))
    
    # updating the PN coefficients as uL_d_uS1, etc, may have changed
    aa, bb = get_PN_coeff(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2)
#     gg, gl = get_gg_gl_from_aa_bb(aa, bb)
    
#     # get GW evolution of the frequency
#     inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)

    inv_dMw_dt_M = 1./get_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast_N4LO(M_omega, y_nat_vect, par_circ, par_simp_prec, 
                          al_offset=0):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))
    
    # updating the PN coefficients as uL_d_uS1, etc, may have changed
    aa, bb = get_PN_coeff(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2)
#     gg, gl = get_gg_gl_from_aa_bb(aa, bb)
    
#     # get GW evolution of the frequency
#     inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)

    inv_dMw_dt_M = 1./get_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast_N4LO(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
#     print(M_omega, dy_dMw)
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast_fix_PN_coeff(M_omega, y_nat_vect, par_circ, par_simp_prec, 
                                       aa, bb,
                                       al_offset=0):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))

    # get GW evolution of the frequency
    inv_dMw_dt_M = 1./get_inv_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast_fix_PN_coeff_alt(M_omega, y_nat_vect, par_circ, par_simp_prec, 
                                       gg, gl,
                                       al_offset=0):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))

    # get GW evolution of the frequency
    inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast_N4LO_fix_PN_coeff(
                        M_omega, y_nat_vect, par_circ, par_simp_prec, 
                        aa, bb, 
                        al_offset=0):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))
    
    # get GW evolution of the frequency
    inv_dMw_dt_M = 1./get_dMw_dt_M(M_omega_pows, aa, bb)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast_N4LO(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
    return dy_dMw


@njit(fastmath=True, cache=True)
def evol_binary_circ_fast_N4LO_fix_PN_coeff_alt(
                        M_omega, y_nat_vect, par_circ, par_simp_prec, 
                        gg, gl, 
                        al_offset=0):
    # parse parameters
    # 0-2
    # 3-5
    # 6-8
    
    uL_v = y_nat_vect[0:3]
    uS1_v = y_nat_vect[3:6]
    uS2_v = y_nat_vect[6:9]
    
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    # scalar quantities that will be useful for the other parts
    M_omega_1_3 = M_omega**(0.333333333)
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get alpha from simple precession
    al_0, dal_0_dMw = get_al_dal_0(M_omega_pows, par_simp_prec,
                                   al_offset)
    
    # do not need to go to the inertial frame !
#     uL_v = rot_z(al_0, uL_r_v)
#     uS1_v = rot_z(al_0, uS1_r_v)
#     uS2_v = rot_z(al_0, uS2_r_v)
    
    uJ_0_v = np.array([0., 0., 1.])
    
    uL_d_uS1 = inner(uL_v, uS1_v)
    uL_d_uS2 = inner(uL_v, uS2_v)
    uS1_d_uS2 = inner(uS1_v, uS2_v)
    
    uL_c_uS1_v = cross(uL_v, uS1_v)
    uL_c_uS2_v = cross(uL_v, uS2_v)
    uS1_c_uS2_v = cross(uS1_v, uS2_v)
    
    uJ0_c_uL_v = cross(uJ_0_v, uL_v)
    uJ0_c_uS1_v = cross(uJ_0_v, uS1_v)
    uJ0_c_uS2_v = cross(uJ_0_v, uS2_v)
    
    cross_prods=np.vstack((uL_c_uS1_v, 
                           uL_c_uS2_v, 
                           uS1_c_uS2_v, 
                           uJ0_c_uL_v, 
                           uJ0_c_uS1_v, 
                           uJ0_c_uS2_v))
    
    # get GW evolution of the frequency
    inv_dMw_dt_M = get_inv_dMw_dt_M(M_omega_pows, gg, gl)
    
    # get precession
    par_SP_fast = np.array([
        inv_L_N_Mt, inv_dMw_dt_M, 
        uL_d_uS1, uL_d_uS2, uS1_d_uS2, 
        al_0, dal_0_dMw])
    dy_dMw = get_dy_dMw_SP_fast_N4LO(M_omega_pows, cross_prods, par_circ, par_SP_fast)
    
    return dy_dMw

    
########


@njit(fastmath=True, cache=True)
def get_dMw_dt_M_from_M_omega_sequence(M_omega, aa, bb):
    # scalars
    M_omega_1_3 = M_omega**0.33333333
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_3_3 = M_omega
    M_omega_4_3 = M_omega * M_omega_1_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    
    log_M_omega = np.log(M_omega)
    a0, a2, a3, a4, a5, a6, a7, a8 \
        = aa[0], aa[1], aa[2], aa[3], aa[4], aa[5], aa[6], aa[7]
    b6, b8 \
        = bb[0], bb[1]    
    dMw_dt_M = M_omega_6_3 * a0 * M_omega_5_3\
            * (1. \
               + a2 * M_omega_2_3 + a3 * M_omega + a4 * M_omega_4_3\
               + a5 * M_omega_5_3 + (a6 + b6 * log_M_omega) * M_omega_6_3\
               + a7 * M_omega_7_3\
               + (a8 + b8 * log_M_omega) * M_omega_8_3)
    return dMw_dt_M    


@njit(fastmath=True, cache=True)
def get_uLS12_i_from_M_omega_sequence(M_omega, uL_r_v, uS1_r_v, uS2_r_v, 
                                      par_circ, par_simp_prec, 
                                      al_offset=0.):
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    A_n3, A_n2, A_n1 \
            = par_simp_prec
    
    #
    M_omega_1_3 = M_omega**0.333333333333
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_3_3 = M_omega
    
    # get the leading order azimuth angle
    al_0 = A_n3 / M_omega_3_3 + A_n2 / M_omega_2_3 + A_n1 / M_omega_1_3
    al_0 += al_offset
    
    ca, sa = np.cos(al_0), np.sin(al_0)
    
    #
    n_pt = len(M_omega_1_3)
    
    uL_i_v = np.zeros((3, n_pt))
    uS1_i_v = np.zeros((3, n_pt))
    uS2_i_v = np.zeros((3, n_pt))
    
    uL_i_v[0, :] = ca * uL_r_v[0, :] - sa * uL_r_v[1, :]
    uL_i_v[1, :] = sa * uL_r_v[0, :] + ca * uL_r_v[1, :]
    
    uS1_i_v[0, :] = ca * uS1_r_v[0, :] - sa * uS1_r_v[1, :]
    uS1_i_v[1, :] = sa * uS1_r_v[0, :] + ca * uS1_r_v[1, :]
    
    uS2_i_v[0, :] = ca * uS2_r_v[0, :] - sa * uS2_r_v[1, :]
    uS2_i_v[1, :] = sa * uS2_r_v[0, :] + ca * uS2_r_v[1, :]
    
    return uL_i_v, uS1_i_v, uS2_i_v
    
@njit(fastmath=True, cache=True)    
def get_al_0_from_M_omega_sequence(M_omega_pows, eta, par_simp_prec, 
                                   al_offset=0.):
    A_n3, A_n2, A_n1 \
            = par_simp_prec
    
    #
    M_omega_1_3 = M_omega_pows[0, :]
    M_omega_2_3 = M_omega_pows[1, :]
    M_omega_3_3 = M_omega_pows[2, :]
    
    # get the leading order azimuth angle
    al_0 = A_n3 / M_omega_3_3 + A_n2 / M_omega_2_3 + A_n1 / M_omega_1_3
    al_0 += al_offset
    return al_0

@njit(cache=True)
def get_L_pn_v_amp_only_from_scalar_f(f_gw, M1, M2, L_v, S1_v, S2_v):
    """
    FIXME
    check consistency with the precession equation
    should make it flexible in PN order
    """
    
    Mt = M1+M2
    delta = (M1-M2)/(M1+M2)
    eta = (M1*M2)/Mt**2.
    eta2 = eta*eta
    eta3 = eta*eta2
    
    S_Mt = G*Mt**2./c
    
    Mt_omega = G*(M1+M2)*(np.pi*f_gw)/c**3
    Mt_omega_2_3 = Mt_omega**(2./3.)
    Mt_omega_4_3 = Mt_omega_2_3 * Mt_omega_2_3
    Mt_omega_5_3 = Mt_omega_2_3 * Mt_omega
    Mt_omega_2 = Mt_omega * Mt_omega
#     Mt_omega_7_3 = Mt_omega * Mt_omega_4_3
    
    Lx, Ly, Lz = L_v[0], L_v[1], L_v[2]
    S1x, S1y, S1z = S1_v[0], S1_v[1], S1_v[2]
    S2x, S2y, S2z = S2_v[0], S2_v[1], S2_v[2]

    # note L is not necessarily along z!
    # a projection is needed in general
    L = np.sqrt(Lx**2. + Ly**2. + Lz**2.)
    S1_L = (Lx * S1x + Ly * S1y + Lz * S1z)/L    
    S2_L = (Lx * S2x + Ly * S2y + Lz * S2z)/L    
    
    SS_z_Mt2 = (S1_L + S2_L) / S_Mt    
    Sig_z_Mt2 = (M1 + M2) * (S2_L/M2 - S1_L/M1) / S_Mt    
        
    # Newtonian
    L = np.sqrt(Lx**2. + Ly**2 + Lz**2)
    
    PN1p0 = eta/6 + 1.5
    PN1p5 = -35./6. * SS_z_Mt2 - 2.5 * delta * Sig_z_Mt2
    PN2p0 = eta2/24. - 19.*eta/8. + 27./8.
    PN2p5 = (-77./8. + 427./72.*eta) * SS_z_Mt2 \
            + delta * (-21./8. + 35.*eta/12.) * Sig_z_Mt2
    PN3p0 = 7.*eta3/1296 + 31.*eta2/24 + (41.*np.pi**2. - 6889/144)*eta + 135./16.    
#     PN3p5 = (-405./16.+1101./16.*eta-29./16.*eta2) * SS_z_Mt2\
#             + delta * (-81./6.+117./4.*eta-15./16.*eta2) * Sig_z_Mt2
    
    PN_tot = (1. + PN1p0 * Mt_omega_2_3 + PN1p5 * Mt_omega \
                 + PN2p0 * Mt_omega_4_3 + PN2p5 * Mt_omega_5_3\
                 + PN3p0 * Mt_omega_2\
             )   
    
    # consistent with Pratten+ 21, include only amplitude changes to L
    L_pn_x = Lx * PN_tot
    L_pn_y = Ly * PN_tot
    L_pn_z = Lz * PN_tot
    
    return L_pn_x, L_pn_y, L_pn_z

@njit(cache=True)
def get_L_pn_v_from_scalar_f(f_gw, M1, M2, L_v, S1_v, S2_v):
    """
    FIXME
    check consistency with the precession equation
    should make it flexible in PN order
    """
    
    Mt = M1+M2
    delta = (M1-M2)/(M1+M2)
    eta = (M1*M2)/Mt**2.
    eta2 = eta*eta
    eta3 = eta*eta2
    
    Mt_M1 = Mt/M1
    Mt_M2 = Mt/M2
    
    S_Mt = G*Mt**2./c
    
    Mt_omega = G*(M1+M2)*(np.pi*f_gw)/c**3
    Mt_omega_2_3 = Mt_omega**(2./3.)
    Mt_omega_4_3 = Mt_omega_2_3 * Mt_omega_2_3
    Mt_omega_5_3 = Mt_omega_2_3 * Mt_omega
    Mt_omega_2 = Mt_omega * Mt_omega
#     Mt_omega_7_3 = Mt_omega * Mt_omega_4_3
    
    Lx, Ly, Lz = L_v[0], L_v[1], L_v[2]
    S1x, S1y, S1z = S1_v[0], S1_v[1], S1_v[2]
    S2x, S2y, S2z = S2_v[0], S2_v[1], S2_v[2]

    # note L is not necessarily along z!
    # a projection is needed in general
    L = np.sqrt(Lx**2. + Ly**2. + Lz**2.)
    S1_L = (Lx * S1x + Ly * S1y + Lz * S1z)/L    
    S2_L = (Lx * S2x + Ly * S2y + Lz * S2z)/L    
    
    SS_z_Mt2 = (S1_L + S2_L) / S_Mt    
    Sig_z_Mt2 = (M1 + M2) * (S2_L/M2 - S1_L/M1) / S_Mt    
        
    # Newtonian
    L = np.sqrt(Lx**2. + Ly**2 + Lz**2)
    
    PN1p0 = eta/6 + 1.5
    PN1p5 = -35./6. * SS_z_Mt2 - 2.5 * delta * Sig_z_Mt2
    PN2p0 = eta2/24. - 19.*eta/8. + 27./8.
    PN2p5 = (-77./8. + 427./72.*eta) * SS_z_Mt2 \
            + delta * (-21./8. + 35.*eta/12.) * Sig_z_Mt2
    PN3p0 = 7.*eta3/1296 + 31.*eta2/24 + (41.*np.pi**2. - 6889/144)*eta + 135./16.    
#     PN3p5 = (-405./16.+1101./16.*eta-29./16.*eta2) * SS_z_Mt2\
#             + delta * (-81./6.+117./4.*eta-15./16.*eta2) * Sig_z_Mt2
    
    PN_tot = (1. + PN1p0 * Mt_omega_2_3 + PN1p5 * Mt_omega \
                 + PN2p0 * Mt_omega_4_3 + PN2p5 * Mt_omega_5_3\
                 + PN3p0 * Mt_omega_2\
             )   
    
    # consistent with Pratten+ 21, include only amplitude changes to L
    L_pn_x = Lx * PN_tot
    L_pn_y = Ly * PN_tot
    L_pn_z = Lz * PN_tot
    
    
    
    # now add in corrections to the direction of L
    L_pn_v = np.array([L_pn_x, L_pn_y, L_pn_z])
    
    PN1p5_S_v = - (0.25*Mt_M1 + 0.75) * S1_v\
                - (0.25*Mt_M2 + 0.75) * S2_v
    PN2p5_S_v = (0.4375 * Mt_M1 - 1.9375 + eta * (0.45833333333333 * Mt_M1 + 0.1875)) * S1_v\
              + (0.4375 * Mt_M2 - 1.9375 + eta * (0.45833333333333 * Mt_M1 + 0.1875)) * S2_v
    
    L_pn_v += eta * (Mt_omega_2_3 * PN1p5_S_v + Mt_omega_4_3 * PN2p5_S_v)
    
    L_pn_x, L_pn_y, L_pn_z = L_pn_v[0], L_pn_v[1], L_pn_v[2]
    
    return L_pn_x, L_pn_y, L_pn_z


@njit(fastmath=True, cache=True)
def get_PN_coeff(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2):
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    eta = M1*M2/(M1+M2)**2.
    eta_2 = eta * eta
    eta_3 = eta_2 * eta
    eta_4 = eta_3 * eta
    
    M1_Mt_sq = (M1/Mt) * (M1/Mt)
    M2_Mt_sq = (M2/Mt) * (M2/Mt)
    
    # exact expression    
#     a0 = (96./5.) * eta_i
    
#     beta3 = ( chi1 * uLi_d_uS1 * (113.*M1_Mt_sq + 75.*eta_i) \
#              +chi2 * uLi_d_uS2 * (113.*M2_Mt_sq + 75.*eta_i) )/12.
#     sigma4 = eta_i*chi1*chi2*(247.*uS1_d_uS2 - 721.*uLi_d_uS1*uLi_d_uS2)/48. \
#            + (  M1_Mt_sq * chi1**2. * (233.-719.*uLi_d_uS1**2.) \
#               + M2_Mt_sq * chi2**2. * (233.-719.*uLi_d_uS2**2.))/96.
#     beta5 = M1_Mt_sq * chi1 * ((31319./1008.-1159*eta_i/24.) + qq * (809./84.-281.*eta_i/8)) * uLi_d_uS1\
#           + M2_Mt_sq * chi2 * ((31319./1008.-1159*eta_i/24.) + 1./qq * (809./84.-281.*eta_i/8)) * uLi_d_uS2
#     beta6 = np.pi * M1_Mt_sq * chi1 * (37.5 + 151./6. * qq) * uLi_d_uS1\
#           + np.pi * M2_Mt_sq * chi2 * (37.5 + 151./6. / qq) * uLi_d_uS2
#     beta7 = M1_Mt_sq * chi1 * ((130325./756. - 796069./2016.*eta_i + 100019./864.*eta_2) \
#                                + (1195759./18144. - 257023./1008.*eta_i + 2903./32.*eta_2) * qq) * uLi_d_uS1 \
#           + M2_Mt_sq * chi2 * ((130325./756. - 796069./2016.*eta_i + 100019./864.*eta_2) \
#                                + (1195759./18144. - 257023./1008.*eta_i + 2903./32.*eta_2) / qq) * uLi_d_uS2 
#     beta_8 = np.pi * M1_Mt_sq * chi1 * ((76927/504.-220055/672.*eta_i) \
#                                         + (1665./28-50483./224*eta_i) * qq) * uLi_d_uS1\
#             +np.pi * M2_Mt_sq * chi2 * ((76927/504.-220055/672.*eta_i) \
#                                         + (1665./28-50483./224*eta_i) / qq) * uLi_d_uS2
    
#     a2 = - (743.+924.*eta_i) / 336.
#     a3 = 4.*np.pi - beta3
#     a4 = 34103./18144. + 13661./2016.*eta_i + 59./18.*eta_2 - sigma4
#     a5 = - np.pi*(4159. + 15876.*eta_i)/672. - beta5
#     a6 = 16447322263./139708800. + 16*np.pi**2./3. - 856./105.*np.log(16.) - 1712./105.*gamma_E - beta6\
#        + (451.*np.pi**2./48.-273811877./1088640.) * eta\
#        + 541./896. * eta_2 - 5605./2592. * eta_3
#     a7 = np.pi*( -4415./4032. + 358675./6048.*eta_i + 91495./1512.*eta_2) - beta7
#     a8 = 3971984677513./25427001600. + 127751.*np.log(2)/1470. - 47385.*np.log(3)/1568. \
#         + 124741*gamma_E/4410. -361./126*np.pi**2.\
#         + (82651980013./838252800 -1712.*np.log(2) /315- 856.*gamma_E/315-31495.*np.pi**2./8064)*eta_i\
#         + (54732199./93312-3157.*np.pi**2./144.)*eta_2\
#         - 18927373./435456*eta_3 -95.*eta_2*eta_2/3888 - beta_8
#     a9 = 343801320119*np.pi/745113600. - 13696.*np.pi*np.log(2)/105. - 6848*np.pi*gamma_E/105.\
#         + (-51438847*np.pi/48384 + 205./6.*np.pi**3. )*eta_i \
#         + (42680611*np.pi/145152)*eta_2 + 9731*np.pi/1344.*eta_3
    
    
#     b6 = -1712./315.
#     b8 = -856./315. * eta_i + 124741./4410.
#     b9 = -6848/105.*np.pi


    # numerical prefactors evaluated
    a0 = 19.2 * eta 
    
    beta3 = ( chi1 * uL_d_uS1 * (9.416666666666667*M1_Mt_sq + 6.25*eta) \
            + chi2 * uL_d_uS2 * (9.416666666666667*M2_Mt_sq + 6.25*eta) )
    sigma4 = eta*chi1*chi2*(5.145833333333333*uS1_d_uS2 - 15.020833333333334*uL_d_uS1*uL_d_uS2) \
           + (  M1_Mt_sq * chi1*chi1 * (2.427083333333333-7.489583333333333*uL_d_uS1*uL_d_uS1) \
              + M2_Mt_sq * chi2*chi2 * (2.427083333333333-7.489583333333333*uL_d_uS2*uL_d_uS2))
    beta5 = M1_Mt_sq*chi1*((31.07043650793651-48.29166666666667*eta) +    qq*(9.630952380952381-35.125*eta))*uL_d_uS1\
          + M2_Mt_sq*chi2*((31.07043650793651-48.29166666666667*eta) + 1./qq*(9.630952380952381-35.125*eta))*uL_d_uS2
    beta6 = np.pi * M1_Mt_sq * chi1 * (37.5 + 25.16666666666667 * qq) * uL_d_uS1\
          + np.pi * M2_Mt_sq * chi2 * (37.5 + 25.16666666666667 / qq) * uL_d_uS2
    beta7 = M1_Mt_sq * chi1 * ((172.38756613756613 - 394.875496031746*eta + 115.76273148148148*eta_2) \
                             + (65.90382495590829 - 254.9831349206349*eta + 90.71875*eta_2) * qq) * uL_d_uS1 \
          + M2_Mt_sq * chi2 * ((172.38756613756613 - 394.875496031746*eta + 115.76273148148148*eta_2) \
                             + (65.90382495590829 - 254.9831349206349*eta + 90.71875*eta_2) / qq) * uL_d_uS2 
    beta_8 = M1_Mt_sq * chi1 * ((479.5105120291706 -1028.7547193239614*eta) \
                                        + (186.81256315096448 - 708.022419335596*eta) * qq) * uL_d_uS1\
           + M2_Mt_sq * chi2 * ((479.5105120291706 -1028.7547193239614*eta) \
                                        + (186.81256315096448 - 708.022419335596*eta) / qq) * uL_d_uS2    
    
    a2 = - (2.2113095238095237+2.75*eta) 
    a3 = 12.566370614359172 - beta3
    a4 = 1.8795745149911816 + 6.776289682539683*eta + 3.2777777777777778*eta_2 - sigma4
    a5 = - (19.443279533154687 + 74.22012644105887*eta) - beta5
    a6 = 138.34906951952667 - beta6\
       - 158.78421870835658 * eta\
       + 0.6037946428571429 * eta_2 - 2.162422839506173 * eta_3
    a7 = ( -3.440012789087038+ 186.31130043424588*eta + 190.10583322764424*eta_2) - beta7
#     a8 = 171.2993474851841 + 54.71751970726321 * eta + 370.17311782978675*eta_2 \
#         - 43.46563831937096 * eta_3 - 0.024434156378600823 * eta_4 - beta_8

    a8=0.
    
    b6 = -5.434920634920635 
#     b8 = 28.285941043083902 - 2.7174603174603176 * eta 
    
    b8=0.
    
    aa = np.array([a0, a2, a3, a4, a5, a6, a7, a8])
    bb = np.array([b6, b8])
    return aa, bb

@njit(fastmath=True, cache=True)
def get_PN_coeff_sequence(par_circ, uL_d_uS1, uL_d_uS2, uS1_d_uS2):
    # par
    M1, M2, Mt, qq, eta, \
    t_Mt, \
    chi1, chi2, S1_Mt, S2_Mt\
            = par_circ
    
    eta = M1*M2/(M1+M2)**2.
    eta_2 = eta * eta
    eta_3 = eta_2 * eta
    eta_4 = eta_3 * eta
    
    M1_Mt_sq = (M1/Mt) * (M1/Mt)
    M2_Mt_sq = (M2/Mt) * (M2/Mt)
    
    n_pt = len(uL_d_uS1)
    
    # numerical prefactors evaluated
    a0 = 19.2 * eta * np.ones(n_pt)
    
    beta3 = ( chi1 * uL_d_uS1 * (9.416666666666667*M1_Mt_sq + 6.25*eta) \
            + chi2 * uL_d_uS2 * (9.416666666666667*M2_Mt_sq + 6.25*eta) )
    sigma4 = eta*chi1*chi2*(5.145833333333333*uS1_d_uS2 - 15.020833333333334*uL_d_uS1*uL_d_uS2) \
           + (  M1_Mt_sq * chi1*chi1 * (2.427083333333333-7.489583333333333*uL_d_uS1*uL_d_uS1) \
              + M2_Mt_sq * chi2*chi2 * (2.427083333333333-7.489583333333333*uL_d_uS2*uL_d_uS2))
    beta5 = M1_Mt_sq*chi1*((31.07043650793651-48.29166666666667*eta) +    qq*(9.630952380952381-35.125*eta))*uL_d_uS1\
          + M2_Mt_sq*chi2*((31.07043650793651-48.29166666666667*eta) + 1./qq*(9.630952380952381-35.125*eta))*uL_d_uS2
    beta6 = np.pi * M1_Mt_sq * chi1 * (37.5 + 25.16666666666667 * qq) * uL_d_uS1\
          + np.pi * M2_Mt_sq * chi2 * (37.5 + 25.16666666666667 / qq) * uL_d_uS2
    beta7 = M1_Mt_sq * chi1 * ((172.38756613756613 - 394.875496031746*eta + 115.76273148148148*eta_2) \
                             + (65.90382495590829 - 254.9831349206349*eta + 90.71875*eta_2) * qq) * uL_d_uS1 \
          + M2_Mt_sq * chi2 * ((172.38756613756613 - 394.875496031746*eta + 115.76273148148148*eta_2) \
                             + (65.90382495590829 - 254.9831349206349*eta + 90.71875*eta_2) / qq) * uL_d_uS2 
    beta_8 = M1_Mt_sq * chi1 * ((479.5105120291706 -1028.7547193239614*eta) \
                                        + (186.81256315096448 - 708.022419335596*eta) * qq) * uL_d_uS1\
           + M2_Mt_sq * chi2 * ((479.5105120291706 -1028.7547193239614*eta) \
                                        + (186.81256315096448 - 708.022419335596*eta) / qq) * uL_d_uS2    
    
    a2 = - (2.2113095238095237+2.75*eta) * np.ones(n_pt)
    a3 = 12.566370614359172 - beta3
    a4 = 1.8795745149911816 + 6.776289682539683*eta + 3.2777777777777778*eta_2 - sigma4
    a5 = - (19.443279533154687 + 74.22012644105887*eta) - beta5
    a6 = 138.34906951952667 - beta6\
       - 158.78421870835658 * eta\
       + 0.6037946428571429 * eta_2 - 2.162422839506173 * eta_3
    a7 = ( -3.440012789087038+ 186.31130043424588*eta + 190.10583322764424*eta_2) - beta7
#     a8 = 171.2993474851841 + 54.71751970726321 * eta + 370.17311782978675*eta_2 \
#         - 43.46563831937096 * eta_3 - 0.024434156378600823 * eta_4 - beta_8

    a8=np.zeros(n_pt)
    
    b6 = -5.434920634920635 * np.ones(n_pt)    
#     b8 = 28.285941043083902 - 2.7174603174603176 * eta * np.ones(n_pt)

    b8=np.zeros(n_pt)
    
    aa = np.vstack((a0, a2, a3, a4, a5, a6, a7, a8))
    bb = np.vstack((b6, b8))
    return aa, bb

@njit(cache=True)
def get_gg_gl_from_aa_bb(aa, bb):
    a0, a2, a3, a4, a5, a6, a7, a8 \
        = aa[0], aa[1], aa[2], aa[3], aa[4], aa[5], aa[6], aa[7]
    b6, b8 = bb[0], bb[1]
    
    a2_2 = a2*a2
    
    g0 = 1.
    g2 = -a2
    g3 = -a3
    g4 = - (a4 - a2_2)
    g5 = - (a5 - 2.*a3*a2)
    g6 = - (a6 - 2.*a4*a2 - a3*a3 + a2_2*a2)
    g7 = - (a7 - 2.*a5*a2 - 2.*a4*a3 + 3.*a3*a2_2)
    g8 = - (a8 - 2.*a6*a2 - 2.*a5*a3 - a4*a4 + 3.*a4*a2_2 + 3.*a3*a3*a2 - a2_2*a2_2)
    
    # I think the 3 is double-counted
    gl6 = -b6 
    gl8 = -(b8 - 2.*a2*b6)
    
    gg = np.array([g0, g2, g3, g4, g5, g6, g7, g8]) / a0
    gl = np.array([gl6, gl8]) / a0 

    return gg, gl

@njit(fastmath=True, cache=True)
def get_al_coeff_NNLO(M1, M2, chiL, chip):
    """
    LALsimIMRPhenomX_precession.c lines 808-831
    multiple differences with the paper PRD 103, 104056 (2021)???
    """
    Mt = M1+M2
    
    eta = M1*M2/(M1+M2)**2.
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    eta6 = eta5 * eta
    
    delta = (M1-M2)/(M1+M2)
    delta2 = delta*delta
    delta3 = delta2 * delta
    
    m1   = M1/Mt
    m1_2 = m1 * m1
    m1_3 = m1_2 * m1
    m1_4 = m1_3 * m1
    m1_5 = m1_4 * m1
    m1_6 = m1_5 * m1
    m1_7 = m1_6 * m1
    m1_8 = m1_7 * m1
    
    chiL2 = chiL * chiL
    chip2 = chip * chip
    
    A_n3    = -35/192. + (5*delta)/(64.*m1)
    
    A_n2    = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta
    
    A_n1    = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)\
              + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)\
              + ((15*chip2*delta*m1_3)/128. - (35*chip2*m1_4)/128.)/eta2
    

    A_log   = ((5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.\
             + ((-2035*chiL*delta*m1)/21504.\
             + (2995*chiL*m1_2)/9216.)/eta + ((5*chiL*chip2*delta*m1_5)/128.\
             - (35*chiL*chip2*m1_6)/384.)/eta3\
             - (35*np.pi)/48. + (5*delta*np.pi)/(16.*m1))
    
    A_p1    = (5*(-190512*delta3*eta6 + 2268*delta2*eta3*m1*(eta2*(323 + 784*eta)\
             + 336*(25*chiL2 + chip2)*m1_4) + 7*m1_3*(8024297*eta4 + 857412*eta5\
             + 3080448*eta6 + 143640*chip2*eta2*m1_4\
             - 127008*chip2*(-4*chiL2 + chip2)*m1_8\
             + 6048*eta3*((2632*chiL2 + 115*chip2)*m1_4 - 672*chiL*m1_2*np.pi))\
             + 3*delta*m1_2*(-5579177*eta4 + 80136*eta5 - 3845520*eta6\
             + 146664*chip2*eta2*m1_4 + 127008*chip2*(-4*chiL2 + chip2)*m1_8\
             - 42336*eta3*((726*chiL2 + 29*chip2)*m1_4\
             - 96*chiL*m1_2*np.pi))))/(6.5028096e7*eta4*m1_3)
    
    return A_n3, A_n2, A_n1, A_log, A_p1
    
    

########

@njit(cache=True)
def get_eul_diff_core(M_omega, uL_r_v, 
                 eta, par_simp_prec, 
                 c_beta_0,
                 al_offset=0.):
    """
    assume uL_r_v is in a frame aligned with J(0) and precess at al_0
    """
    A_n3, A_n2, A_n1 \
            = par_simp_prec
    
    #
    M_omega_1_3 = M_omega**0.33333333333
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # get the leading order azimuth angle
    al_0 = A_n3 / M_omega + A_n2 / M_omega_2_3 + A_n1 / M_omega_1_3
    al_0 += al_offset
    
    ep_0 = c_beta_0 * al_0
    
    c_beta = uL_r_v[2]
#     beta = np.arccos(c_beta)
    beta = np.real(np.arccos(c_beta + 0j))
    
    # get the corrections
    al_1_wrap = np.arctan2(uL_r_v[1], uL_r_v[0])
    return al_0, al_1_wrap, beta, c_beta, ep_0
    
    
def get_eul_diff(M_omega, uL_r_v, 
                 eta, par_simp_prec, 
                 c_beta_0,
                 al_offset=0.):
    al_0, al_1_wrap, beta, c_beta, ep_0 \
        = get_eul_diff_core(M_omega, uL_r_v, 
                            eta, par_simp_prec, 
                            c_beta_0, al_offset)
    al_1 = np.unwrap(al_1_wrap)
    al = al_0 + al_1
    ep_1 = integ.cumulative_trapezoid(c_beta - c_beta_0, al, initial=0)\
           + c_beta_0 * al_1
    return al_1, beta, ep_1, al_0, ep_0


@njit(cache=True)
def find_chi_L_v(uL_v, chi1_v, chi2_v, 
                 uN_v=np.array([1, 0, 0])):
    zz = uL_v
    
    if inner(uL_v-uN_v, uL_v-uN_v) > 1e-9:
        yy = cross(uL_v, uN_v)
    elif inner(uL_v-np.array([1, 0, 0]), uL_v-np.array([1, 0, 0])) > 1e-9:
        yy = cross(uL_v, np.array([1, 0, 0]))
    else:
        yy = cross(uL_v, np.array([0, 1, 0]))    
    yy = yy /np.sqrt(inner(yy, yy))
    xx = cross(yy, zz)
    xx = xx/np.sqrt(inner(xx, xx))    
    
    chi1_L_v = np.array([
        inner(chi1_v, xx), 
        inner(chi1_v, yy),
        inner(chi1_v, zz)
    ]) 
    
    chi2_L_v = np.array([
        inner(chi2_v, xx), 
        inner(chi2_v, yy),
        inner(chi2_v, zz)
    ])
    return chi1_L_v, chi2_L_v


def kwargs_at_f_ref_new_simple(f_ref_new, **kwargs):    
    
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    
    f_ref = kwargs['f_ref']
    
    phi_ref = kwargs['phi_ref']
    iota = kwargs['iota']
    
    
    ############################
    # parameters of the system #
    ############################ 
    
    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    qq = M2/M1
    
    Mt = M1 + M2
    mu = M1 * M2 / Mt
    eta = mu / Mt    
    
    M1_2 = M1 * M1
    M2_2 = M2 * M2
    Mt_2 = Mt * Mt
    
    GMt = G*Mt
    G_c = G/c
    
    M1_Mt_2 = M1_2 / Mt_2
    M2_Mt_2 = M2_2 / Mt_2
    
    chi1 = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2 = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    S1 = chi1 * G_c * M1_2
    S2 = chi2 * G_c * M2_2
    
    r_Mt = G_c*Mt/c
    t_Mt = r_Mt/c
    t_Mt_pi = t_Mt * np.pi
    
    S_Mt = G_c*Mt_2
    
    S1_Mt = S1/S_Mt
    S2_Mt = S2/S_Mt
    
    chiL = (M1 * chi1z + M2 * chi2z) / Mt
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
    
    S_p = chip * G_c * M1_2
    S_l = chi1z * G_c * M1_2 + chi2z * G_c * M2_2
    
    A_n3, A_n2, A_n1, A_log, A_p1 = get_al_coeff_NNLO(M1, M2, chiL, chip)
    
    f_gw_init = f_ref
    omega_init = np.pi*f_gw_init
    a_init = (GMt/omega_init**2.)**(1/3.)
    L_N_init = mu * np.sqrt(GMt*a_init)

    M_omega = t_Mt * omega_init
    M_omega_1_3 = M_omega**0.3333333333
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # first set the coordinate such that L is along z
    L_N_v_init = np.array([0, 0, 1]) * L_N_init
    
    S1_v_init = np.array([
        chi1x, 
        chi1y, 
        chi1z]) * G*M1**2./c
    S2_v_init = np.array([
        chi2x, 
        chi2y, 
        chi2z]) * G*M2**2./c
    
    # line of sight in L0 frame
    # Eq (35c) of T1500606-v6; 
    # LALSimIMRPhenomP.c XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame
    N_v = np.array([
        np.sin(iota) * np.cos(np.pi/2. - phi_ref), 
        np.sin(iota) * np.sin(np.pi/2. - phi_ref), 
        np.cos(iota)
    ])
    
    # get PN corrections
    # XPHM includes 3 PN corrections to the magnitude only
    L_PN_x, L_PN_y, L_PN_z = get_L_pn_v_amp_only_from_scalar_f(f_gw_init, M1, M2, 
                                             L_N_v_init, S1_v_init, S2_v_init)    
    
    L_PN_v_init = np.array([L_PN_x, L_PN_y, L_PN_z])
    L_PN_init = np.sqrt(inner(L_PN_v_init, L_PN_v_init))
        
    J_PN_v_init = L_PN_v_init + S1_v_init + S2_v_init
    J_PN_init = np.sqrt(inner(J_PN_v_init, J_PN_v_init))
    
    uJ_v = J_PN_v_init / J_PN_init
    uL_v = L_PN_v_init / L_PN_init
    
    # for anti-aligned sys, flip uJ to be consistent with its value at r->inf
    if (L_PN_z + S1_v_init[2] + S2_v_init[2]) < 0:
        uJ_v = - uJ_v
        al_ex = np.pi
    else:
        al_ex = 0.
    
    # construct a new coordinate 
    # s.t. z is along J and N is in the x-z plane 
        
    if inner(uJ_v-N_v, uJ_v-N_v)>1e-9:
        zz = uJ_v
        yy = cross(uJ_v, N_v)
        yy = yy /np.sqrt(inner(yy, yy))
        xx = cross(yy, zz)
        xx = xx/np.sqrt(inner(xx, xx))
        
    elif inner(uL_v-N_v, uL_v-N_v)>1e-9:
        zz = N_v
        yy = cross(N_v, uL_v)
        yy = yy /np.sqrt(inner(yy, yy))
        xx = cross(yy, zz)
        xx = xx/np.sqrt(inner(xx, xx))
    
    else:
        zz = np.array([0, 0, 1])
        xx = np.array([1, 0, 0])
        yy = np.array([0, 1, 0])
    
    # now translates to this new coordiate
    L_N_v_init = np.array([
        inner(L_N_v_init, xx), 
        inner(L_N_v_init, yy), 
        inner(L_N_v_init, zz), 
    ])
    L_PN_v_init = np.array([
        inner(L_PN_v_init, xx), 
        inner(L_PN_v_init, yy), 
        inner(L_PN_v_init, zz), 
    ])
    N_v = np.array([
        inner(N_v, xx),
        inner(N_v, yy),
        inner(N_v, zz),
    ])
    
    ##############################
    # how much has L_N precessed #
    ##############################
    
    
    par_simp_p = np.array([A_n3, A_n2, A_n1])

    al_offset, __ = get_al_dal_0(M_omega_pows, par_simp_p)
    al_offset *= -1
    
    M_omega_new = t_Mt * f_ref_new
    M_omega_new_1_3 = M_omega_new**0.3333333333
    M_omega_new_2_3 = M_omega_new_1_3 * M_omega_new_1_3
    M_omega_new_4_3 = M_omega_new_2_3 * M_omega_new_2_3
    M_omega_new_5_3 = M_omega_new_4_3 * M_omega_new_1_3
    M_omega_new_6_3 = M_omega_new_5_3 * M_omega_new_1_3
    M_omega_new_7_3 = M_omega_new_6_3 * M_omega_new_1_3
    M_omega_new_8_3 = M_omega_new_7_3 * M_omega_new_1_3
    M_omega_new_9_3 = M_omega_new_8_3 * M_omega_new_1_3
    
    M_omega_new_pows = np.array([M_omega_new_1_3, M_omega_new_2_3, M_omega_new, 
                                 M_omega_new_4_3, M_omega_new_5_3, M_omega_new_6_3, 
                                 M_omega_new_7_3, M_omega_new_8_3, M_omega_new_9_3])    

    inv_L_N_new_Mt = M_omega_new_1_3 / eta
    L_N_new = S_Mt/inv_L_N_new_Mt 
    
    # alpha angle at the new f_ref
    al_new, __ = get_al_dal_0(M_omega_new_pows, par_simp_p)
    al_new += al_offset + al_ex
    
    c_beta_new = (L_N_new + S_l) / np.sqrt((L_N_new + S_l)**2. + S_p**2.)
    beta_new = np.arccos(c_beta_new)
    
    uL_N_new = np.array([
        np.sin(beta_new) * np.cos(al_new),
        np.sin(beta_new) * np.sin(al_new),
        np.cos(beta_new)        
    ])
    
    
    iota_new = np.arccos(inner(uL_N_new, N_v))    
    
    kwargs_out = kwargs.copy()
    kwargs_out['iota'] = iota_new
    kwargs_out['f_ref'] = f_ref_new
    
    return kwargs_out
    

def wrap_ODE_fast_Euler_only(return_ode_pts=False, include_tail=True, 
                             use_N4LO_prec=True,
                             fix_PN_coeff=False, 
                             **kwargs):
    """
    Wrapper for functions computing Euler angles vs. orbital frequency (including GW tail corrections)
    Outputs:
        eul_vs_Mw: Given M*omega, returns alpha, cos(beta/2), sin(beta/2), epsilon
        th_JN_0: angle between total angular momentum J_0 and line of sight N
        pol: 2*zeta, see eqs. (c22), (D6), & (D7) of Pratten+ 21, PRD 103, 104056
        chi1_L_v: chi1 vector at 6M or end of the ODE in a frame aligned with L
        chi2_L_v: chi2 vector at 6M or end of the ODE in a frame aligned with L
        
    Options for the user to choose:
        return_ode_pts: return points used to evaluate the precession ODE. Default: False
        include_tail: assuming the user input 2*np.pi*f_gw/|m'|, either treat it as orbital frequency (include_tail=False) or include further corretions due to GW tail (include_tail=True). Default: True
        use_N4LO_prec: If True, solve N4LO precession equation. Otherwise solve the NLO equations based on which the MSA angles are derived. Default: True
        fix_PN_coeff: If True, use chi1z and chi2z at f_ref when computing domega/dt. Otherwise update them together with precession. Default: False
    """
    
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    
    f_ref = kwargs['f_ref']
    f_lower = kwargs['f_lower']
    
    f_lower = np.amin([f_lower, f_ref])
    
#     make f_lower lower than requested
    f_lower *= 0.99
#     f_lower *=0.495
    
    phi_ref = kwargs['phi_ref']
    iota = kwargs['iota']
    
    atol = kwargs.pop('atol', 2e-4)
    rtol = kwargs.pop('rtol', 2e-4)
    max_abs_m = kwargs.pop('max_abs_m', 4)
    
    if max_abs_m <2:
        max_abs_m = 2
    
    ############################
    # parameters of the system #
    ############################ 
    
    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    qq = M2/M1
    
    Mt = M1 + M2
    mu = M1 * M2 / Mt
    eta = mu / Mt
    
    M1_2 = M1 * M1
    M2_2 = M2 * M2
    Mt_2 = Mt * Mt
    
    GMt = G*Mt
    G_c = G/c
    
    M1_Mt_2 = M1_2 / Mt_2
    M2_Mt_2 = M2_2 / Mt_2
    
    chi1 = np.sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2 = np.sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    S1 = chi1 * G_c * M1_2
    S2 = chi2 * G_c * M2_2
    
    r_Mt = G_c*Mt/c
    t_Mt = r_Mt/c
    t_Mt_pi = t_Mt * np.pi
    
    S_Mt = G_c*Mt_2
    
    S1_Mt = S1/S_Mt
    S2_Mt = S2/S_Mt
    
    chiL = (M1 * chi1z + M2 * chi2z) / Mt
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
    
    A_n3, A_n2, A_n1, A_log, A_p1 = get_al_coeff_NNLO(M1, M2, chiL, chip)
    
    # isco
    f_isco = np.sqrt(GMt/(6.*r_Mt)**3.)/np.pi
    
    # to make sure that at least we have an inspiral part
    if f_lower> (0.8*f_isco):
        f_lower = 0.8*f_isco
    
    # max freq to integrate the ode
    omega_max = np.pi * f_isco
    
    # FIXME
    # explore if we have improvements when evolving beyond isco
    omega_max *= 2.5
#     omega_max *= 1.2
    
    Mw_max = t_Mt * omega_max

    f_gw_init = f_ref
    omega_init = np.pi*f_gw_init
    a_init = (GMt/omega_init**2.)**(1/3.)
    L_N_init = mu * np.sqrt(GMt*a_init)

    M_omega = t_Mt * omega_init
    M_omega_1_3 = M_omega**0.3333333333
    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
    
    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, M_omega, 
                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
    inv_L_N_Mt = M_omega_1_3 / eta
    
    # first set the coordinate such that L is along z
    L_N_v_init = np.array([0, 0, 1]) * L_N_init
    
    S1_v_init = np.array([
        chi1x, 
        chi1y, 
        chi1z]) * G*M1**2./c
    S2_v_init = np.array([
        chi2x, 
        chi2y, 
        chi2z]) * G*M2**2./c
    
    J_N_v_init = L_N_v_init + S1_v_init + S2_v_init
    J_N_init = np.sqrt(inner(J_N_v_init, J_N_v_init))
    
    # line of sight in L0 frame
    # Eq (35c) of T1500606-v6; 
    # LALSimIMRPhenomP.c XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame
    N_v = np.array([
        np.sin(iota) * np.cos(np.pi/2. - phi_ref), 
        np.sin(iota) * np.sin(np.pi/2. - phi_ref), 
        np.cos(iota)
    ])
    
    # defines polarization
    # see eq. c24 in XPHM paper
    # in L0 frame
    xN_v = np.array([
        -np.cos(iota) * np.sin(phi_ref),
        -np.cos(iota) * np.cos(phi_ref),
        np.sin(iota)
    ])
    
    # get PN corrections
    # XPHM includes 3 PN corrections to the magnitude only
#     L_PN_x, L_PN_y, L_PN_z = get_L_pn_v_amp_only_from_scalar_f(f_gw_init, M1, M2, 
#                                              L_N_v_init, S1_v_init, S2_v_init)
    
#     # we add in also directional corrections at 1.5 and 2.5 PN
    L_PN_x, L_PN_y, L_PN_z = get_L_pn_v_from_scalar_f(f_gw_init, M1, M2, 
                                             L_N_v_init, S1_v_init, S2_v_init)
    
    
    L_PN_v_init = np.array([L_PN_x, L_PN_y, L_PN_z])
    L_PN_init = np.sqrt(inner(L_PN_v_init, L_PN_v_init))
        
    J_PN_v_init = L_PN_v_init + S1_v_init + S2_v_init
    J_PN_init = np.sqrt(inner(J_PN_v_init, J_PN_v_init))
    
    uJ_v = J_PN_v_init / J_PN_init
    uL_v = L_PN_v_init / L_PN_init
    
    th_JN_0 = np.arccos(inner(N_v, uJ_v))
    
    # eq. c7 in the XPHM paper
    # used to determine the epsilon offset
    if chip > 1e-9:
        phi_JL0 = np.arctan2(uJ_v[1], uJ_v[0])
    else:
        phi_JL0 = 0.
    
    # construct a new coordinate 
    # s.t. z is along J and N is in the x-z plane 
        
    if inner(uJ_v-N_v, uJ_v-N_v)>1e-9:
        zz = uJ_v
        yy = cross(uJ_v, N_v)
        yy = yy /np.sqrt(inner(yy, yy))
        xx = cross(yy, zz)
        xx = xx/np.sqrt(inner(xx, xx))
        
    elif inner(uL_v-N_v, uL_v-N_v)>1e-9:
        zz = N_v
        yy = cross(N_v, uL_v)
        yy = yy /np.sqrt(inner(yy, yy))
        xx = cross(yy, zz)
        xx = xx/np.sqrt(inner(xx, xx))
    
    else:
        zz = np.array([0, 0, 1])
        xx = np.array([1, 0, 0])
        yy = np.array([0, 1, 0])
    
    # now translates to this new coordiate
    L_N_v_init = np.array([
        inner(L_N_v_init, xx), 
        inner(L_N_v_init, yy), 
        inner(L_N_v_init, zz), 
    ])
    L_PN_v_init = np.array([
        inner(L_PN_v_init, xx), 
        inner(L_PN_v_init, yy), 
        inner(L_PN_v_init, zz), 
    ])
    S1_v_init = np.array([
        inner(S1_v_init, xx), 
        inner(S1_v_init, yy), 
        inner(S1_v_init, zz), 
    ])
    S2_v_init = np.array([
        inner(S2_v_init, xx), 
        inner(S2_v_init, yy), 
        inner(S2_v_init, zz), 
    ])
    N_v = np.array([
        inner(N_v, xx),
        inner(N_v, yy),
        inner(N_v, zz),
    ])
    xN_v = np.array([
        inner(xN_v, xx),
        inner(xN_v, yy),
        inner(xN_v, zz)
    ])

    
    # in the ODE we need the Newtonian L!
    uL_N_v_init = L_N_v_init / L_N_init
    
    if chi1>1e-9:
        uS1_v_init = S1_v_init / S1
    else:
        uS1_v_init = np.array([0, 0, 1])
    if chi2>1e-9:
        uS2_v_init = S2_v_init / S2
    else:
        uS2_v_init = np.array([0, 0, 1])
    
    cb_init = uL_N_v_init[2]
    
    # polarization angle
    # see LALSimIMRPhenomX_precession.c
    # lines 710 - 785
    # P & Q are x'_N and y'_N in eq. c23; note in the paper a typo in y'_N
    P_pol_v = np.array([N_v[2], 0, -N_v[0]])
    Q_pol_v = np.array([0, 1, 0])
    X_d_P = inner(xN_v, P_pol_v)
    X_d_Q = inner(xN_v, Q_pol_v)
    
    # here we include the factor of 2
    # so that pol = 2*zeta, 
    # and hp =  hp_0 * cos(pol) + hc_0 * sin(pol)
    #     hc = -hp_0 * sin(pol) + hc_0 * cos(pol)
    # see LALSimIMRPhenomXPHM.c, IMRPhenomXPHM_hplushcross()
    pol = 2.*np.arctan2(X_d_Q, X_d_P)
    
    ######################
    # parameters for ODE #
    ######################
    
    par = np.array([M1, M2, Mt, qq, eta, 
                    t_Mt, 
                    chi1, chi2, S1_Mt, S2_Mt])

    # for fast precession
    par_simp_p = np.array([A_n3, A_n2, A_n1])

    al_offset, __ = get_al_dal_0(M_omega_pows, par_simp_p)
    al_offset *= -1

    # at f_ref, the two frames coincides
    y_nat_init = np.hstack((
        uL_N_v_init,
        uS1_v_init,
        uS2_v_init
    ))
    
    aa_ref, bb_ref = get_PN_coeff(par, 
                                  inner(uL_N_v_init, uS1_v_init), 
                                  inner(uL_N_v_init, uS2_v_init), 
                                  inner(uS1_v_init, uS2_v_init))
#     gg_ref, gl_ref = get_gg_gl_from_aa_bb(aa_ref, bb_ref)
    
    dMw_dt_M_ref = get_dMw_dt_M(M_omega_pows, aa_ref, bb_ref)
    forward_flag=True
    
    if ((f_ref>f_isco) and (dMw_dt_M_ref < (1e-6*eta))) or (3.16*f_ref >= omega_max):
        dMw_dt_M_ref = 1e-6*eta
        forward_flag=False                
    
    #######
    # ODE #
    #######
    
    resolution_flag=False
    n_iter = 0
    
    while (not resolution_flag) and (n_iter<3):
        
        # function to do the integration
        if use_N4LO_prec:
            if fix_PN_coeff:
                int_func = lambda Mw_, y_nat_vect_: \
                    evol_binary_circ_fast_N4LO_fix_PN_coeff(
                            Mw_, y_nat_vect_, par, par_simp_p, aa_ref, bb_ref,
                            al_offset=al_offset)
            else:
                int_func = lambda Mw_, y_nat_vect_: \
                    evol_binary_circ_fast_N4LO(Mw_, y_nat_vect_, par, par_simp_p, al_offset=al_offset)
        else:
            if fix_PN_coeff:
                int_func = lambda Mw_, y_nat_vect_: \
                    evol_binary_circ_fast_fix_PN_coeff(
                            Mw_, y_nat_vect_, par, par_simp_p, aa_ref, bb_ref, 
                            al_offset=al_offset)
            else:
                int_func = lambda Mw_, y_nat_vect_: \
                    evol_binary_circ_fast(Mw_, y_nat_vect_, par, par_simp_p, al_offset=al_offset)
        
        def event_6M(Mw_, y_nat_vect_):
            resi = Mw_ - 0.06804138
            return resi
        
        event_6M.direction = 1
        event_6M.terminal = False
        
        if fix_PN_coeff:
            def event_dMw_zero(Mw_, y_nat_vect_):
                if Mw_<0.04:
                    resi = 1.
                else:
                    # scalar quantities that will be useful for the other parts
                    M_omega_1_3 = Mw_**(0.333333333)
                    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
                    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
                    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
                    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
                    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
                    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
                    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
                    
                    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, Mw_, 
                                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                                             M_omega_7_3, M_omega_8_3, M_omega_9_3])            
    
                    dMw_dt_M = get_dMw_dt_M(M_omega_pows, aa_ref, bb_ref)
                    
                    resi = (dMw_dt_M - 0.8 * dMw_dt_M_ref) * 1e6
                return resi
        else:
            def event_dMw_zero(Mw_, y_nat_vect_):
                if Mw_<0.04:
                    resi = 1.
                else:
                    uL_v = y_nat_vect_[0:3]
                    uS1_v = y_nat_vect_[3:6]
                    uS2_v = y_nat_vect_[6:9]
            
                    # scalar quantities that will be useful for the other parts
                    M_omega_1_3 = Mw_**(0.333333333)
                    M_omega_2_3 = M_omega_1_3 * M_omega_1_3
                    M_omega_4_3 = M_omega_2_3 * M_omega_2_3
                    M_omega_5_3 = M_omega_4_3 * M_omega_1_3
                    M_omega_6_3 = M_omega_5_3 * M_omega_1_3
                    M_omega_7_3 = M_omega_6_3 * M_omega_1_3
                    M_omega_8_3 = M_omega_7_3 * M_omega_1_3
                    M_omega_9_3 = M_omega_8_3 * M_omega_1_3
                    
                    M_omega_pows = np.array([M_omega_1_3, M_omega_2_3, Mw_, 
                                             M_omega_4_3, M_omega_5_3, M_omega_6_3, 
                                             M_omega_7_3, M_omega_8_3, M_omega_9_3])
                
                    uL_d_uS1 = inner(uL_v, uS1_v)
                    uL_d_uS2 = inner(uL_v, uS2_v)
                    uS1_d_uS2 = inner(uS1_v, uS2_v)
                
                    # updating the PN coefficients as uL_d_uS1, etc, may have changed
                    aa, bb = get_PN_coeff(par, uL_d_uS1, uL_d_uS2, uS1_d_uS2)
                    
                    dMw_dt_M = get_dMw_dt_M(M_omega_pows, aa, bb)
                    
                    resi = (dMw_dt_M - 0.8 * dMw_dt_M_ref) * 1e6
                return resi
        
        event_dMw_zero.direction = -1
        event_dMw_zero.terminal = True                    
        
        # forward integration
        if forward_flag:
            sol_f = integ.solve_ivp(int_func, \
                    t_span=(f_ref*t_Mt_pi, omega_max*t_Mt), y0=y_nat_init, rtol=rtol, atol=atol, 
                            events=[event_6M, event_dMw_zero])
            
            Mw_f = sol_f.t    
            y_f = sol_f.y
            
        else:
            Mw_f, y_f = np.zeros(0), np.zeros((9, 0))
                
    
        # backward integration
    #     print('sol_b')
        sol_b = integ.solve_ivp(int_func, \
                t_span=(f_ref*t_Mt_pi, f_lower*t_Mt_pi), y0=y_nat_init, rtol=2*rtol, atol=2*atol, 
                        first_step = 0.1*(f_ref-f_lower)*t_Mt_pi)
        
        Mw_b = sol_b.t
        y_b = sol_b.y
        Mw_b = Mw_b[::-1]
        y_b = y_b[:, ::-1]
        n_back = len(Mw_b)    
        
        # backward integration part 2
        # for the |m|>2 harmonic 
        
        if (max_abs_m > 2):
    #         print('sol_b2')
            sol_b2 = integ.solve_ivp(int_func, \
                    t_span=(f_lower*t_Mt_pi, (2./max_abs_m) * f_lower*t_Mt_pi), 
                            y0=sol_b.y[:, -1], rtol=4*rtol, atol=4*atol, 
                            first_step = 0.2*(1.-2./max_abs_m)*f_lower*t_Mt_pi)                                        
        
            Mw_b2 = sol_b2.t
            y_b2 = sol_b2.y
            Mw_b2 = Mw_b2[::-1]
            y_b2 = y_b2[:, ::-1]
            n_back2 = len(Mw_b2)
            
            if forward_flag:
                M_omega = np.hstack((Mw_b2[0:n_back2-1], Mw_b[0:n_back-1], Mw_f)) 
                y = np.hstack((y_b2[:, 0:n_back2-1], y_b[:, 0:n_back-1], y_f))        
                idx_ref = n_back2 + n_back - 2
            else:
                M_omega = np.hstack((Mw_b2[0:n_back2-1], Mw_b)) 
                y = np.hstack((y_b2[:, 0:n_back2-1], y_b))        
                idx_ref = len(M_omega) - 1
            
        else:               
            if forward_flag:
                M_omega = np.hstack((Mw_b[0:n_back-1], Mw_f)) 
                y = np.hstack((y_b[:, 0:n_back-1], y_f))
                idx_ref = n_back - 1
            else:
                M_omega = Mw_b
                y = y_b
                idx_ref = len(M_omega)-1
        
        if idx_ref<0:
            idx_ref = 0
            
            
        idx = M_omega < (t_Mt * omega_max)
            
        # request at least 6 steps in the ODE to enable interpolation
        if len(M_omega[idx])>5:
            resolution_flag=True
            
        else:
            # solve the ODE again with a finer resolution
            atol /= 5
            rtol /= 5
            n_iter += 1        
        
    ###########################
    # processing the solution #
    ###########################    
    
    uL_r_v = y[0:3, :]
    
    al_1, beta, ep_1, \
    al_0, ep_0 = get_eul_diff(M_omega, uL_r_v, 
                              eta, par_simp_p, 
                              cb_init,
                              al_offset=al_offset)
    
    c_hb = np.cos(beta/2.)
    s_hb = np.sin(beta/2.)
    
    # using convention 1 according to the XPHM paper; see their table IV 
#     al_offset = -(al_0 + al_1)[idx_ref]
    al_offset = 0.
    ep_offset = -(ep_0 + ep_1)[idx_ref] + phi_JL0 - np.pi
    
    
    # now select only points within omega_max
    M_omega = M_omega[idx]
    y = y[:, idx]
    al_0 = al_0[idx]
    al_1 = al_1[idx]
    c_hb = c_hb[idx]
    s_hb = s_hb[idx]
    ep_0 = ep_0[idx]
    ep_1 = ep_1[idx]        
    
    # output
    if include_tail:
        # this includes the GW tail contribution to the phase
        # only a small correction, therefore just fix PN coefficients by their initial values
        dMw_dt_M = get_dMw_dt_M_from_M_omega_sequence(M_omega, aa_ref, bb_ref)
                
        Mw_2_3 = M_omega**0.66666666667
        xx = M_omega \
            - dMw_dt_M * (  (2.-5./3.*eta*Mw_2_3)*np.log(M_omega)\
                                  + (2.-eta*Mw_2_3) )        
    else:
        xx = M_omega 
        
    if len(xx)<=3:
        al_vs_Mw_tck = interp.splrep(xx, al_0+al_1+al_offset, k=1)
        c_hb_vs_Mw_tck = interp.splrep(xx, c_hb, k=1)
        s_hb_vs_Mw_tck = interp.splrep(xx, s_hb, k=1)
        ep_vs_Mw_tck = interp.splrep(xx, ep_0+ep_1 + ep_offset, k=1)
    else:
        al_vs_Mw_tck = interp.splrep(xx, al_0+al_1+al_offset, k=3)
        c_hb_vs_Mw_tck = interp.splrep(xx, c_hb, k=3)
        s_hb_vs_Mw_tck = interp.splrep(xx, s_hb, k=3)
        ep_vs_Mw_tck = interp.splrep(xx, ep_0+ep_1 + ep_offset, k=3)
    
    def eul_vs_Mw(_Mw):
        _Mw = np.asarray(_Mw)
        _al = interp.splev(_Mw, al_vs_Mw_tck, ext=3)
        _c_hb = interp.splev(_Mw, c_hb_vs_Mw_tck, ext=3)
        _s_hb = interp.splev(_Mw, s_hb_vs_Mw_tck, ext=3)
        _ep = interp.splev(_Mw, ep_vs_Mw_tck, ext=3)
        return _al, _c_hb, _s_hb, _ep
    
    
    # spin in L frame at 6M
    if forward_flag:
        y_6M = np.squeeze(sol_f.y_events[0])
        if not len(y_6M)>0:
            # dMw=0 before binary separation decreases to 6M
            # use the last point
            y_6M = y[:, -1]
            
    else:
        y_6M = y[:, -1]
        
    chi1_L_v, chi2_L_v = find_chi_L_v(
                        y_6M[0:3], chi1*y_6M[3:6], chi2*y_6M[6:9], 
                        N_v)
    
    chi1_L_mag = np.sqrt(inner(chi1_L_v, chi1_L_v))
    if chi1_L_mag>0.9999:
        chi1_L_v *= (0.9999 / chi1_L_mag)
        
    chi2_L_mag = np.sqrt(inner(chi2_L_v, chi2_L_v))
    if chi2_L_mag>0.9999:
        chi2_L_v *= (0.9999 / chi2_L_mag)    
    
    if return_ode_pts:
        return eul_vs_Mw, th_JN_0, pol, \
               chi1_L_v, chi2_L_v, \
               M_omega/t_Mt, al_0, al_1, \
               beta, ep_0, ep_1
    else:
        return eul_vs_Mw, th_JN_0, pol, \
               chi1_L_v, chi2_L_v