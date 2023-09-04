"""
    Codes to generate the XODE waveform.  
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
from numba import jit, njit, prange

import lal
import lalsimulation as lals

from .LAL_constants import *
from . import precLib as pLib


######################
# top-level wrappers #
######################

def get_hp_hc_f_sequence(approximant, 
                         ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
                         mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
                         **kwargs):    
    """
    Wrapper function to produce hp(f) and hc(f). 
    Inputs:
        approximant: XODE or XPHM. Decides which approximant to call. 
        freqs: frequency grid in [Hz] on which the waveform is computed. It can have arbitrary spacing but every element needs to be positive
        ll_list_neg: l quantum numbers of the coprecessing modes. It needs to have the same size as mm_list_neg
        mm_list_neg: m quantum numbers of the coprecessing modes. Only negative m's have support for positive frequencies. 
        mass1: Mass of the heavier BH in [solar mass]
        mass2: Mass of the lighter BH in [solar mass]
        spin1x: x component of BH1's dimensionless spin 
        spin1y: y component of BH1's dimensionless spin
        spin1z: z component of BH1's dimensionless spin (aligned with LN)
        spin2x: x component of BH2's dimensionless spin 
        spin2y: y component of BH2's dimensionless spin
        spin2z: z component of BH2's dimensionless spin (aligned with LN)
        distance: distance measured in [Mpc]
        iota: inclination (angle between LN and N)
        phi_ref: reference phase so that Nx = \sin \iota * \cos (\pi/2 - \phi_ref) in the L_0 phase. 
        f_ref: reference GW frequency in [Hz]. For meaningful result, it should be smaller than the ISCO frequency (though the code may still run if a higher f_ref is given). 
        atol: atol to be passed to the ODE solver. 
        rtol: rtol to be passed to the ODE solver. 
        aux_par: lal.CreateDict() if approximant = 'XODE'. If approximant = 'XPHM', use it to pass none-default XPHM options.
        
    There are also a few options for the users to turn on/off:
        SEOB_22_cal: [default True] if True, recalibrate the co-precessing 22 mode's phase to match SEOBNRv4PHM
        SEOB_HM_cal: [default True] if True, recalibrate the co-precessing 21 & 33 modes' phase to match SEOBNRv4PHM
        use_N4LO_prec: [default True] if True, use N4LO precession equations following Akcay+ 2021, PRD 103, 024014. If False, use NLO equations to be consistent with the MSA construction.
        fix_PN_coeff: [default False] if True, use spin1z and spin2z defined at f_ref to compute d\omega/dt in the precession. Otherwise, updating spin1z and spin2z together with the precession equations. Default to False. 
           
        
    Outputs:
        hp(f) and hc(f). The conventions should follow the default of XPHM. 
    """
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    
    if M1_Ms < M2_Ms:
        M1_Ms, M2_Ms = M2_Ms, M1_Ms
        chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = \
        chi2x, chi2y, chi2z, chi1x, chi1y, chi1z
        
        kwargs['mass1'], kwargs['mass2'] = M1_Ms, M2_Ms
        kwargs['spin1x'], kwargs['spin1y'], kwargs['spin1z'] = chi1x, chi1y, chi1z
        kwargs['spin2x'], kwargs['spin2y'], kwargs['spin2z'] = chi2x, chi2y, chi2z 
    
    qq = M2_Ms/M1_Ms
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
        
    if (approximant==1) or ('XODE' in approximant):
        if chip > 1e-6:
            # use XODE only when the system is precessing
            hp, hc = get_hp_hc_f_dmn_XPHM_f_sequence(ll_list_neg, mm_list_neg, 
                                **kwargs)
        else:
            # for aligned system, simply query XHM
            # note XPHM is problematic for anti-aligned system, so it is not used here
            lal_keys_hh =[
                         'freqs',
                         'm1_kg', 'm2_kg',
                         's1z', 's2z',
                         'dist', 'iota', 'phi_ref', 'f_ref',
                         'aux_par'
                         ]
            lal_par_dict = {
                'freqs': lal.CreateREAL8Sequence(len(kwargs['freqs'])),
                'm1_kg': kwargs['mass1']*Ms,
                'm2_kg': kwargs['mass2']*Ms,
                's1z': kwargs['spin1z'],
                's2z': kwargs['spin2z'],
                'dist':kwargs['distance']*1e6*pc,
                'iota':kwargs['iota'],
                'phi_ref':kwargs['phi_ref'],
                'f_ref':kwargs['f_ref'],
                'aux_par': kwargs.pop('aux_par', lal.CreateDict())
            }
            lal_par_dict['freqs'].data = kwargs['freqs']
            
            mm_list = np.hstack((mm_list_neg, -mm_list_neg))
            ll_list = np.hstack((ll_list_neg, ll_list_neg))
            mode_array = lals.SimInspiralCreateModeArray()
            for i in range(len(mm_list)):
                lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list[i]), int(mm_list[i]))
            lals.SimInspiralWaveformParamsInsertModeArray(lal_par_dict['aux_par'], mode_array)
            
            kwargs['aux_par'] = lal_par_dict['aux_par']
            
            hp, hc = lals.SimIMRPhenomXHMFrequencySequence(*[lal_par_dict[key] for key in lal_keys_hh])
            hp = hp.data.data
            hc = hc.data.data
            
    else:
        lal_keys_hh =[
                     'freqs',
                     'm1_kg', 'm2_kg',
                     's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
                     'dist', 'iota', 'phi_ref', 'f_ref',
                     'aux_par'
                     ]
        lal_par_dict = {
            'freqs': lal.CreateREAL8Sequence(len(kwargs['freqs'])),
            'm1_kg': kwargs['mass1']*Ms,
            'm2_kg': kwargs['mass2']*Ms,
            's1x': kwargs['spin1x'], 
            's1y': kwargs['spin1y'],
            's1z': kwargs['spin1z'],
            's2x': kwargs['spin2x'],
            's2y': kwargs['spin2y'],
            's2z': kwargs['spin2z'],
            'dist':kwargs['distance']*1e6*pc,
            'iota':kwargs['iota'],
            'phi_ref':kwargs['phi_ref'],
            'f_ref':kwargs['f_ref'],
            'aux_par': kwargs.pop('aux_par', lal.CreateDict())
        }
        lal_par_dict['freqs'].data = kwargs['freqs']
        
        mm_list = np.hstack((mm_list_neg, -mm_list_neg))
        ll_list = np.hstack((ll_list_neg, ll_list_neg))
        mode_array = lals.SimInspiralCreateModeArray()
        for i in range(len(mm_list)):
            lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list[i]), int(mm_list[i]))
        lals.SimInspiralWaveformParamsInsertModeArray(lal_par_dict['aux_par'], mode_array)
        
        kwargs['aux_par'] = lal_par_dict['aux_par']
        
        hp, hc = lals.SimIMRPhenomXPHMFrequencySequence(*[lal_par_dict[key] for key in lal_keys_hh])
        hp = hp.data.data
        hc = hc.data.data
        
    return hp, hc



def get_hp_hc_each_prec_mode_f_sequence(
                         ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
                         mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
                         **kwargs):    
    """
    Similar to get_hp_hc_f_sequence()
    but now return h_pc_each_mode with shape (2, len(mm_list_neg), n_freq)
    i.e., each coprecessing mode's contribution to the final hp & hc
    
    Assumes XODE is used. 
    """
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    
    if M1_Ms < M2_Ms:
        M1_Ms, M2_Ms = M2_Ms, M1_Ms
        chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = \
        chi2x, chi2y, chi2z, chi1x, chi1y, chi1z
        
        kwargs['mass1'], kwargs['mass2'] = M1_Ms, M2_Ms
        kwargs['spin1x'], kwargs['spin1y'], kwargs['spin1z'] = chi1x, chi1y, chi1z
        kwargs['spin2x'], kwargs['spin2y'], kwargs['spin2z'] = chi2x, chi2y, chi2z 
    
    qq = M2_Ms/M1_Ms
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
        
    if chip > 1e-6:
        h_pc_each_prec_mode = get_hp_hc_f_dmn_XPHM_f_sequence(ll_list_neg, mm_list_neg, 
                            get_each_prec_mode_contribution = True,
                            **kwargs)
        
    else:
        h_pc_each_prec_mode = np.zeros( (2, len(mm_list_neg), len(kwargs['freqs'])), dtype=np.complex64)
        lal_keys_hh =[
                     'freqs',
                     'm1_kg', 'm2_kg',
                     's1z', 's2z',
                     'dist', 'iota', 'phi_ref', 'f_ref',
                     'aux_par'
                     ]
        lal_par_dict = {
            'freqs': lal.CreateREAL8Sequence(len(kwargs['freqs'])),
            'm1_kg': kwargs['mass1']*Ms,
            'm2_kg': kwargs['mass2']*Ms,
            's1z': kwargs['spin1z'],
            's2z': kwargs['spin2z'],
            'dist':kwargs['distance']*1e6*pc,
            'iota':kwargs['iota'],
            'phi_ref':kwargs['phi_ref'],
            'f_ref':kwargs['f_ref']
        }
        lal_par_dict['freqs'].data = kwargs['freqs']
        
        for i in range(len(mm_list_neg)):
            lal_par_dict['aux_par'] = lal.CreateDict()
            mode_array = lals.SimInspiralCreateModeArray()
            lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list_neg[i]), int(mm_list_neg[i]))
            lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list_neg[i]), -int(mm_list_neg[i]))
            lals.SimInspiralWaveformParamsInsertModeArray(lal_par_dict['aux_par'], mode_array)
            _hp, _hc = lals.SimIMRPhenomXHMFrequencySequence(*[lal_par_dict[key] for key in lal_keys_hh])
            h_pc_each_prec_mode[0, i, :] = _hp.data.data
            h_pc_each_prec_mode[1, i, :] = _hc.data.data
            
    return h_pc_each_prec_mode


def get_hp_hc_f_low_max(approximant, 
                        ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
                         mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
                         **kwargs):
    """
    similar to get_hp_hc_f_sequence but now computes hp and hc on a frequency grid defined
    between [f_lower, f_max] with a uniform spacing of delta_f
    """
    f_lower = kwargs['f_lower']
    f_max = kwargs['f_max']
    delta_f = kwargs['delta_f']
    freqs = np.arange(f_lower, f_max+0.5*delta_f, delta_f)
    kwargs['freqs'] = freqs
    
    hp, hc = get_hp_hc_f_sequence(approximant, 
                         ll_list_neg, mm_list_neg, 
                         **kwargs)
    return hp, hc


def get_h_lm_inertial_f_sequence_XODE(
        ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
        mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
        **kwargs):
    """
    Wrapper function to produce h_lm(f) in the inertial frame. 
    
    The inputs are the same as those used in get_hp_hc_f_sequence(). Please see the instruction therein. 
    Note: the l & m input via ll_list_neg & mm_list_neg are modes in the COPRECESSING frame!
    
    Outputs: 
        h_lm: modes in the INERTIAL frame. Shape: (n_lm, n_freq),
              where n_lm = 5 * l2_flag + 7 * l3_flag + 9 * l4_flag,
              with l2_flag = 1 if 2 in ll_list_neg, 0 otherwise, and similarly for other flags
        lm_list: (l, m) for each mode in h_lm. Its shape is (n_lm, 2). 
    """
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    
    if M1_Ms < M2_Ms:
        M1_Ms, M2_Ms = M2_Ms, M1_Ms
        chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = \
        chi2x, chi2y, chi2z, chi1x, chi1y, chi1z
        
        kwargs['mass1'], kwargs['mass2'] = M1_Ms, M2_Ms
        kwargs['spin1x'], kwargs['spin1y'], kwargs['spin1z'] = chi1x, chi1y, chi1z
        kwargs['spin2x'], kwargs['spin2y'], kwargs['spin2z'] = chi2x, chi2y, chi2z 
    
    h_lm, lm_list = get_h_lm_inertial_f_sequence_internal(
            ll_list_neg, mm_list_neg, 
            **kwargs)
    return h_lm, lm_list


################################################
# wrappers to generate LAL style waveforms     #
################################################

def get_hp_hc_f_dmn_XPHM_f_sequence(ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
                                    mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
                                    update_spin = False, 
                                    use_N4LO_prec = True,
                                    fix_PN_coeff = False,
                                    SEOB_22_cal = True,
                                    SEOB_HM_cal = True, 
                                    get_each_prec_mode_contribution = False,
                                    **kwargs):
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    dist_Mpc = kwargs['distance']
    iota = kwargs['iota']
    f_ref = kwargs['f_ref']
    phi_ref = kwargs['phi_ref']
    freq_0 = kwargs['freqs']
    delta_f = freq_0[1]-freq_0[0]
    
    f_lower = freq_0[0]
    kwargs['f_lower'] = f_lower
    
    aux_par = kwargs.pop('aux_par', None)
    Mf_cut = kwargs.pop('Mf_cut', 1.0)
    
    if aux_par is None:
        aux_par = lal.CreateDict()        
    
    dist = dist_Mpc * 1e6 * pc
    
    n_pt_0 = len(freq_0)
    n_mode_h = len(mm_list_neg)
    n_mode = n_mode_h * 2
    
    max_abs_m = int(np.max(np.abs(mm_list_neg)))
    min_abs_m = int(np.min(np.abs(mm_list_neg)))
    
    ll_list=np.hstack((ll_list_neg, ll_list_neg))
    mm_list=np.hstack((mm_list_neg, -mm_list_neg))
    
    # physical system
    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    qq = M2/M1
    Mt = M1 + M2
    mu = M1 * M2 / Mt
    eta = mu / Mt
    eta2 = eta * eta
    delta = (M1-M2)/Mt
    
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    chi1 = np.sqrt(chi1p**2 + chi1z**2)
    chi2 = np.sqrt(chi2p**2 + chi2z**2)                
    
    chieff = (M1 * chi1z + M2 * chi2z)/Mt
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
    
    r_Mt = G*Mt/c**2.
    t_Mt = r_Mt/c
    # approximate isco by 6M
    f_isco = 0.02165 / t_Mt       
    
    # first get euler angles from ODE    
    all_finite=False
    while not all_finite:
        eul_vs_Mw, th_JN_0, pol, \
        chi1_L_v, chi2_L_v\
            = pLib.wrap_ODE_fast_Euler_only(return_ode_pts=False, 
                                        include_tail=True, 
                                        use_N4LO_prec = use_N4LO_prec, 
                                        fix_PN_coeff = fix_PN_coeff,
                                        max_abs_m = max_abs_m, 
                                        **kwargs)
        Mw_test = np.pi*np.array([freq_0[0], f_isco]) * t_Mt
        _al, _c_hb, _s_hb, _ep = eul_vs_Mw(Mw_test)
        if np.isfinite(_al).all() and np.isfinite(_c_hb).all() and np.isfinite(_ep).all():
            all_finite = True
        else:
            kwargs['atol']*=0.3
            kwargs['rtol']*=0.3
            
    
    # updating chip to value at 6M
    chi12 = pLib.inner(chi1_L_v, chi2_L_v)    
    chi1p = np.sqrt(chi1_L_v[0]**2. + chi1_L_v[1]**2.)
    chi2p = np.sqrt(chi2_L_v[0]**2. + chi2_L_v[1]**2.)
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
        
    
#     only generate waveform at f<f_cut
    f_cut = Mf_cut / t_Mt               
    idx = freq_0 < f_cut
    freq = freq_0[idx]
    
    n_pt = len(freq)
    n_pt_diff = n_pt_0 - n_pt    
    
    
    # note different mp will reach the same orbital frequency at different freq
    # thus the euler angles should be a 2D array; 
    # the first dimension denotes |mp| of the coprecessing modes
    n_eul = max_abs_m - min_abs_m + 1
    al = np.zeros((n_eul, n_pt))
    c_hb = np.zeros((n_eul, n_pt))
    s_hb = np.zeros((n_eul, n_pt))
    ep = np.zeros((n_eul, n_pt))
    
    _mp = min_abs_m
    for i in range(n_eul):
        Mw_w_tail = (2.*np.pi/_mp) * freq * t_Mt
        _al, _c_hb, _s_hb, _ep = eul_vs_Mw(Mw_w_tail)
        al[i, :] = _al
        c_hb[i, :] = _c_hb
        s_hb[i, :] = _s_hb
        ep[i, :] = _ep
        _mp += 1
    
    # get a list of co-precessing modes from lal
    h_lmp = np.zeros((n_mode_h, n_pt), dtype=np.complex64)
    
    mode_array = lals.SimInspiralCreateModeArray()
    for i in range(n_mode):
        lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list[i]), int(mm_list[i]))
    lals.SimInspiralWaveformParamsInsertModeArray(aux_par, mode_array)
    
    # this asks for the co-precessing modes
    lals.SimInspiralWaveformParamsInsertPhenomXPHMPrecModes(aux_par, 1)
    
    lal_keys_hh =[
        'freqs', 'l', 'm',
        'm1_kg', 'm2_kg',
        's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'dist', 'iota', 'phi_ref', 'f_ref',
        'aux_par'
        ]
    
    # note phi_ref = 0 for co-precessing modes
    # this follows XPHM convention 1
    # see table IV of the XPHM paper
    lal_par_dict = {
            'freqs': lal.CreateREAL8Sequence(n_pt),
            'l': 2,
            'm': -2,
            'm1_kg': M1,
            'm2_kg': M2,
            's1x': chi1x, 
            's1y': chi1y,
            's1z': chi1z,
            's2x': chi2x,
            's2y': chi2y,
            's2z': chi2z,
            'dist':dist,
            'iota':iota,
            'phi_ref':0,
            'f_ref':f_ref,
            'aux_par': aux_par,
            }
        
    lal_par_dict['freqs'].data = freq
    
    if update_spin:
        lal_par_dict['s1x'] = chi1_L_v[0]
        lal_par_dict['s1y'] = chi1_L_v[1]
        lal_par_dict['s1z'] = chi1_L_v[2]
        lal_par_dict['s2x'] = chi2_L_v[0]
        lal_par_dict['s2y'] = chi2_L_v[1]
        lal_par_dict['s2z'] = chi2_L_v[2]
    
    for i in range(n_mode_h):
        _ll = ll_list_neg[i]
        _mm = mm_list_neg[i]
    
        lal_par_dict['l'] = int(_ll)
        lal_par_dict['m'] = int(_mm)               
    
        # NOTE: the function changes name & input arguments from lalsimulation v 3.x.x to 5.1.0!!!
        # currently tuned to be consistent with v 5.1.0. 
        # check your lal version in case of error!!!
        _h_pos, __ = lals.SimIMRPhenomXPHMFrequencySequenceOneMode(*[lal_par_dict[key] for key in lal_keys_hh])

        h_lmp[i, :] = _h_pos.data.data
    
        if (chip < 1e-9):
            # phase at f_ref is captured by epsilon in the precessing case
            # in the non-precessing case, need to add ref phase back
            h_lmp[i, :] *= np.exp(-1j*_mm*(phi_ref + np.pi/2))
            
            
            # check me further!
            # seems something wierd in XPHM
            if iota == np.pi:
                h_lmp[i, :] *= np.exp(-1j*(_mm + 2.)*(np.pi/2-phi_ref))
                

    if (SEOB_22_cal or SEOB_HM_cal):  
        chip2 = chip * chip
#         chieff2 = chieff * chieff
                
        AA = np.array([1, eta, eta2, chieff, eta*chieff, eta2*chieff,
                       chip, eta*chip, eta2*chip, 
                       chieff*chip, eta*chieff*chip, 
                       chip2, eta*chip2, eta2*chip2
                       ])        
        h_lmp = coprec_cal(freq, f_ref, t_Mt, 
                           h_lmp, ll_list_neg, mm_list_neg, AA, 
                           SEOB_22_cal, SEOB_HM_cal)                       
        
#     print('h_lmp', h_lmp[:, 0])

    if not get_each_prec_mode_contribution:
        # now do the twisting up
        hp, hc = get_h_iner_pol_from_h_prec_f_dmn(al, c_hb, s_hb, ep, h_lmp, 
                                         ll_list_neg, mm_list_neg,
                                         th_s=th_JN_0, phi_s=0.)
        
        # rotate polarization to be consistent with XPHM
        hp, hc = rot_pol(hp, hc, pol)
        
        # check me further!
        # seems something wierd in XPHM
        if (chip < 1e-9) and (iota == 0):
            hc *= -1
        
        # put hp, hc to the same length as the original freq
        hp = np.hstack((hp, np.zeros(n_pt_diff, dtype=np.complex64)))
        hc = np.hstack((hc, np.zeros(n_pt_diff, dtype=np.complex64)))
        
        return hp, hc
    
    else:
        h_pc_each_prec_mode = get_h_iner_pol_from_h_prec_f_dmn_each_mode(al, c_hb, s_hb, ep, h_lmp, 
                                         ll_list_neg, mm_list_neg,
                                         th_s=th_JN_0, phi_s=0.)
        for i in range(n_mode_h):
            h_pc_each_prec_mode[0, i, :], h_pc_each_prec_mode[1, i, :] \
                = rot_pol(h_pc_each_prec_mode[0, i, :], h_pc_each_prec_mode[1, i, :], pol) 
        
        if (chip < 1e-9) and (iota == 0):
            h_pc_each_prec_mode[1, :, :] *= -1
            
        h_pc_each_prec_mode \
            = np.concatenate((h_pc_each_prec_mode, np.zeros((2, n_mode_h, n_pt_diff), dtype=np.complex64)), axis=-1)
        return h_pc_each_prec_mode
        


def get_h_lm_inertial_f_sequence_internal(ll_list_neg=np.array([ 2,  2,  3,  3,  4]), 
                                    mm_list_neg=np.array([-2, -1, -3, -2, -4]), 
                                    update_spin = False, 
                                    use_N4LO_prec = True,
                                    fix_PN_coeff = False,
                                    SEOB_22_cal = True,
                                    SEOB_HM_cal = True, 
                                    **kwargs):
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1x = kwargs['spin1x']
    chi1y = kwargs['spin1y']
    chi1z = kwargs['spin1z']
    chi2x = kwargs['spin2x']
    chi2y = kwargs['spin2y']
    chi2z = kwargs['spin2z']
    dist_Mpc = kwargs['distance']
    iota = kwargs['iota']
    f_ref = kwargs['f_ref']
    phi_ref = kwargs['phi_ref']
    freq_0 = kwargs['freqs']
    delta_f = freq_0[1]-freq_0[0]
    
    f_lower = freq_0[0]
    kwargs['f_lower'] = f_lower
    
    aux_par = kwargs.pop('aux_par', None)
    Mf_cut = kwargs.pop('Mf_cut', 1.0)
    
    if aux_par is None:
        aux_par = lal.CreateDict()        
    
    dist = dist_Mpc * 1e6 * pc
    
    n_pt_0 = len(freq_0)
    n_mode_h = len(mm_list_neg)
    n_mode = n_mode_h * 2
    
    max_abs_m = int(np.max(np.abs(mm_list_neg)))
    min_abs_m = int(np.min(np.abs(mm_list_neg)))
    
    ll_list=np.hstack((ll_list_neg, ll_list_neg))
    mm_list=np.hstack((mm_list_neg, -mm_list_neg))
    
    # physical system
    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    qq = M2/M1
    Mt = M1 + M2
    mu = M1 * M2 / Mt
    eta = mu / Mt
    eta2 = eta * eta
    delta = (M1-M2)/Mt
    
    chi1p = np.sqrt(chi1x**2. + chi1y**2.)
    chi2p = np.sqrt(chi2x**2. + chi2y**2.)
    
    chi1 = np.sqrt(chi1p**2 + chi1z**2)
    chi2 = np.sqrt(chi2p**2 + chi2z**2)                
    
    chieff = (M1 * chi1z + M2 * chi2z)/Mt
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
    
    r_Mt = G*Mt/c**2.
    t_Mt = r_Mt/c
    # approximate isco by 6M
    f_isco = 0.02165 / t_Mt       
    
    # first get euler angles from ODE    
    all_finite=False
    while not all_finite:
        eul_vs_Mw, th_JN_0, pol, \
        chi1_L_v, chi2_L_v\
            = pLib.wrap_ODE_fast_Euler_only(return_ode_pts=False, 
                                        include_tail=True, 
                                        use_N4LO_prec = use_N4LO_prec, 
                                        fix_PN_coeff = fix_PN_coeff,
                                        max_abs_m = max_abs_m, 
                                        **kwargs)
        Mw_test = np.pi*np.array([freq_0[0], f_isco]) * t_Mt
        _al, _c_hb, _s_hb, _ep = eul_vs_Mw(Mw_test)
        if np.isfinite(_al).all() and np.isfinite(_c_hb).all() and np.isfinite(_ep).all():
            all_finite = True
        else:
            kwargs['atol']*=0.3
            kwargs['rtol']*=0.3
            
    
    # updating chip to value at 6M
    chi12 = pLib.inner(chi1_L_v, chi2_L_v)    
    chi1p = np.sqrt(chi1_L_v[0]**2. + chi1_L_v[1]**2.)
    chi2p = np.sqrt(chi2_L_v[0]**2. + chi2_L_v[1]**2.)
    chip = 1./(2.+1.5*qq) \
        * np.max(((2.+1.5*qq) * chi1p, (2.+1.5/qq) * chi2p * qq * qq))
        
    
#     only generate waveform at f<f_cut
    f_cut = Mf_cut / t_Mt               
    idx = freq_0 < f_cut
    freq = freq_0[idx]
    
    n_pt = len(freq)
    n_pt_diff = n_pt_0 - n_pt    
    
    
    # note different mp will reach the same orbital frequency at different freq
    # thus the euler angles should be a 2D array; 
    # the first dimension denotes |mp| of the coprecessing modes
    n_eul = max_abs_m - min_abs_m + 1
    al = np.zeros((n_eul, n_pt))
    c_hb = np.zeros((n_eul, n_pt))
    s_hb = np.zeros((n_eul, n_pt))
    ep = np.zeros((n_eul, n_pt))
    
    _mp = min_abs_m
    for i in range(n_eul):
        Mw_w_tail = (2.*np.pi/_mp) * freq * t_Mt
        _al, _c_hb, _s_hb, _ep = eul_vs_Mw(Mw_w_tail)
        al[i, :] = _al
        c_hb[i, :] = _c_hb
        s_hb[i, :] = _s_hb
        ep[i, :] = _ep
        _mp += 1
    
    # get a list of co-precessing modes from lal
    h_lmp = np.zeros((n_mode_h, n_pt), dtype=np.complex64)
    
    mode_array = lals.SimInspiralCreateModeArray()
    for i in range(n_mode):
        lals.SimInspiralModeArrayActivateMode(mode_array, int(ll_list[i]), int(mm_list[i]))
    lals.SimInspiralWaveformParamsInsertModeArray(aux_par, mode_array)
    
    # this asks for the co-precessing modes
    lals.SimInspiralWaveformParamsInsertPhenomXPHMPrecModes(aux_par, 1)
    
    lal_keys_hh =[
        'freqs', 'l', 'm',
        'm1_kg', 'm2_kg',
        's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'dist', 'iota', 'phi_ref', 'f_ref',
        'aux_par'
        ]
    
    # note phi_ref = 0 for co-precessing modes
    # this follows XPHM convention 1
    # see table IV of the XPHM paper
    lal_par_dict = {
            'freqs': lal.CreateREAL8Sequence(n_pt),
            'l': 2,
            'm': -2,
            'm1_kg': M1,
            'm2_kg': M2,
            's1x': chi1x, 
            's1y': chi1y,
            's1z': chi1z,
            's2x': chi2x,
            's2y': chi2y,
            's2z': chi2z,
            'dist':dist,
            'iota':iota,
            'phi_ref':0,
            'f_ref':f_ref,
            'aux_par': aux_par,
            }
        
    lal_par_dict['freqs'].data = freq
    
    if update_spin:
        lal_par_dict['s1x'] = chi1_L_v[0]
        lal_par_dict['s1y'] = chi1_L_v[1]
        lal_par_dict['s1z'] = chi1_L_v[2]
        lal_par_dict['s2x'] = chi2_L_v[0]
        lal_par_dict['s2y'] = chi2_L_v[1]
        lal_par_dict['s2z'] = chi2_L_v[2]
    
    for i in range(n_mode_h):
        _ll = ll_list_neg[i]
        _mm = mm_list_neg[i]
    
        lal_par_dict['l'] = int(_ll)
        lal_par_dict['m'] = int(_mm)               
    
        # NOTE: the function changes name & input arguments from lalsimulation v 3.x.x to 5.1.0!!!
        # currently tuned to be consistent with v 5.1.0. 
        # check your lal version in case of error!!!
        _h_pos, __ = lals.SimIMRPhenomXPHMFrequencySequenceOneMode(*[lal_par_dict[key] for key in lal_keys_hh])

        h_lmp[i, :] = _h_pos.data.data
    
        if (chip < 1e-9):
            # phase at f_ref is captured by epsilon in the precessing case
            # in the non-precessing case, need to add ref phase back
            h_lmp[i, :] *= np.exp(-1j*_mm*(phi_ref + np.pi/2))
            
            
            # check me further!
            # seems something wierd in XPHM
            if iota == np.pi:
                h_lmp[i, :] *= np.exp(-1j*(_mm + 2.)*(np.pi/2-phi_ref))
                

    if (SEOB_22_cal or SEOB_HM_cal):  
        chip2 = chip * chip
#         chieff2 = chieff * chieff
                
        AA = np.array([1, eta, eta2, chieff, eta*chieff, eta2*chieff,
                       chip, eta*chip, eta2*chip, 
                       chieff*chip, eta*chieff*chip, 
                       chip2, eta*chip2, eta2*chip2
                       ])        
        h_lmp = coprec_cal(freq, f_ref, t_Mt, 
                           h_lmp, ll_list_neg, mm_list_neg, AA, 
                           SEOB_22_cal, SEOB_HM_cal)                       
        
    # now do the twisting up
    h_lm, lm_list = get_h_iner_modes_from_h_prec_f_dmn(al, c_hb, s_hb, ep, h_lmp, 
                                     ll_list_neg, mm_list_neg)
    
    # rotate polarization to be consistent with XPHM
    h_lm *= np.exp(1j*pol)
    
    n_lm = h_lm.shape[0]
    
    # put hp, hc to the same length as the original freq
    h_lm = np.concatenate((h_lm, np.zeros((n_lm, n_pt_diff), dtype=np.complex64)), axis=1)    
    return h_lm, lm_list



@njit(fastmath=True, cache=True)
def coprec_cal(freq, f_ref, t_Mt, 
               hlm, ll_list_neg, mm_list_neg, AA, 
               SEOB_22_cal=True, SEOB_HM_cal=True):
    
    chip = AA[6]
    if chip>0.05:
        win = 1.
    else:
        win = 20.*chip
    
    lnMf = np.log(t_Mt*freq)
    n_pt = len(freq)
    
    lnMf2 = lnMf * lnMf
    lnMf3 = lnMf2 * lnMf
    lnMf_pows = np.vstack((lnMf3, lnMf2, lnMf))
    
    lnMf_ref = np.log(t_Mt*f_ref)
    lnMf_ref2 = lnMf_ref * lnMf_ref
    lnMf_ref3 = lnMf_ref2 * lnMf_ref
    lnMf_ref_pows = np.array((lnMf_ref3, lnMf_ref2, lnMf_ref))
    
    n_m = len(mm_list_neg)
    idx_22, idx_33, idx_21 = -1, -1, -1
    for i in range(n_m):
        if int(ll_list_neg[i]) == 2:
            if int(mm_list_neg[i]) == -2:
                idx_22 = i
            elif int(mm_list_neg[i]) == -1:
                idx_21 = i
        elif (int(ll_list_neg[i]) ==3) & (int(mm_list_neg[i])==-3):
            idx_33 = i
            
#     print(idx_22, idx_21, idx_33)
            
    if SEOB_22_cal and (idx_22>=0):
        xx_22 = np.array([[-3.56323838e-02, -3.91799901e-01, -9.75686306e-01],
                          [ 6.88385041e-01,  9.66231641e+00,  4.18895060e+01],
                          [-1.35384639e+00, -1.83658296e+01, -7.72333779e+01],
                          [-2.54389313e-01, -3.80333388e+00, -1.83726453e+01],
                          [ 2.09265319e+00,  3.03359671e+01,  1.38344767e+02],
                          [-4.63153913e+00, -6.63853796e+01, -2.91345765e+02],
                          [-1.13199205e-01, -2.37404309e+00, -1.62808382e+01],
                          [ 2.02797244e+00,  3.74024470e+01,  2.26833885e+02],
                          [-7.76536829e+00, -1.37812021e+02, -7.92163805e+02],
                          [-1.22151881e-01, -1.60476233e+00, -4.48828801e+00],
                          [ 5.91456896e-02, -4.31767946e-01, -1.67819528e+01],
                          [-2.83717558e+00, -5.07668198e+01, -2.87186837e+02],
                          [ 1.88756683e+01,  3.36198656e+02,  1.90689149e+03],
                          [-3.11664903e+01, -5.53809142e+02, -3.16053832e+03]])
                
        _poly_c = AA @ xx_22
        ang_diff_22 = np.zeros(n_pt)
        ang_diff_22_ref = 0
        for k in range(3):
            ang_diff_22 += _poly_c[k] * lnMf_pows[k, :]
            ang_diff_22_ref += _poly_c[k] * lnMf_ref_pows[k]            
        ang_diff_22 = ang_diff_22 - ang_diff_22_ref*np.ones(n_pt)                        
        hlm[idx_22, :] *= np.exp(1j*ang_diff_22*win)
        
    if SEOB_HM_cal:
        if idx_33>=0:
            # 33        
            lnMf_ref = np.log(t_Mt*1.5*f_ref)
            lnMf_ref2 = lnMf_ref * lnMf_ref
            lnMf_ref3 = lnMf_ref2 * lnMf_ref
            lnMf_ref_pows = np.array((lnMf_ref3, lnMf_ref2, lnMf_ref))
            
            xx_33 = np.array([[-5.16817883e-02, -5.93521401e-01, -1.51008151e+00],
                              [ 1.07596885e+00,  1.44729022e+01,  5.64761561e+01],
                              [-3.13015655e+00, -4.27177308e+01, -1.74270627e+02],
                              [ 1.45057669e-01,  2.61181059e+00,  1.41870730e+01],
                              [-2.09505353e+00, -3.66877044e+01, -2.01079809e+02],
                              [ 6.47144670e+00,  1.11454960e+02,  6.11886205e+02],
                              [-3.80841750e-01, -5.80228512e+00, -3.09560763e+01],
                              [ 4.82894930e+00,  7.31911342e+01,  3.77033242e+02],
                              [-1.32011818e+01, -2.02043380e+02, -1.03115056e+03],
                              [-2.85049342e-01, -3.63195235e+00, -1.14999211e+01],
                              [ 3.62108222e-01,  2.76109834e+00, -9.34885212e+00],
                              [-3.11476126e+00, -5.27401530e+01, -2.71698999e+02],
                              [ 2.02101046e+01,  3.40495186e+02,  1.75993806e+03],
                              [-3.35660906e+01, -5.62077689e+02, -2.91600184e+03]])

            _poly_c = AA @ xx_33
            ang_diff_33 = np.zeros(n_pt)
            ang_diff_33_ref = 0
            for k in range(3):
                ang_diff_33 += _poly_c[k] * lnMf_pows[k, :]
                ang_diff_33_ref += _poly_c[k] * lnMf_ref_pows[k]
                
            ang_diff_33 = ang_diff_33 - ang_diff_33_ref*np.ones(n_pt)            
            hlm[idx_33, :] *= np.exp(1j*ang_diff_33*win)
        
        # 21
        if idx_21>=0:
            lnMf_ref = np.log(t_Mt*0.5*f_ref)
            lnMf_ref2 = lnMf_ref * lnMf_ref
            lnMf_ref3 = lnMf_ref2 * lnMf_ref
            lnMf_ref_pows = np.array((lnMf_ref3, lnMf_ref2, lnMf_ref))
            
            xx_21 = np.array([[ 1.45776046e-01,  2.18973977e+00,  1.04486864e+01],
                              [-1.40633005e+00, -2.05836733e+01, -9.54318281e+01],
                              [ 3.46278179e+00,  5.03417892e+01,  2.30135391e+02],
                              [ 1.29800911e-01,  1.78995914e+00,  7.68368363e+00],
                              [-1.56514191e+00, -2.17761304e+01, -9.42715715e+01],
                              [ 3.65947243e+00,  5.02750756e+01,  2.13065232e+02],
                              [-2.36796660e-01, -2.92171503e+00, -1.08633833e+01],
                              [ 3.16000134e+00,  4.03028308e+01,  1.54408897e+02],
                              [-9.47821073e+00, -1.23386514e+02, -4.81713956e+02],
                              [-4.50280739e-01, -6.89852463e+00, -3.36019288e+01],
                              [ 2.18978732e+00,  3.26215732e+01,  1.55407120e+02],
                              [ 5.20528851e-02, -2.57049566e+00, -2.68306710e+01],
                              [-2.11532305e+00, -4.78182689e+00,  1.03293949e+02],
                              [ 8.77287977e+00,  7.28982967e+01,  5.46642926e+01]])      
            
            _poly_c = AA @ xx_21
            ang_diff_21 = np.zeros(n_pt)
            ang_diff_21_ref = 0
            for k in range(3):
                ang_diff_21 += _poly_c[k] * lnMf_pows[k, :]
                ang_diff_21_ref += _poly_c[k] * lnMf_ref_pows[k]
                
            ang_diff_21 = ang_diff_21 - ang_diff_21_ref*np.ones(n_pt)                                            
            hlm[idx_21, :] *= np.exp(1j*ang_diff_21*win)    
        
    return hlm


###########################
# core waveform functions #
###########################

@njit(parallel=True, fastmath=True, cache=True)
def get_h_iner_pol_from_h_prec_f_dmn(al, c_hb, s_hb, ep, h_lmp, 
                                     ll_list_neg, mm_list_neg,
                                     th_s=0., phi_s=0.):
    """
    euler angles (al, beta, ep) have shapes (n_eul, n_freq)
    and the first index corresponds to [abs(mp) - min_abs_m]
    """    
    n_m_neg, n_pt = h_lmp.shape
    min_abs_m = int(np.min(np.abs(mm_list_neg)))
    
    hp = np.zeros(n_pt, dtype=np.complex64)
    hc = np.zeros(n_pt, dtype=np.complex64)
    
    for i in range(n_m_neg):
        _mp = mm_list_neg[i]
        _lp = ll_list_neg[i]

        _idx_eul = int(np.abs(_mp) - min_abs_m)
        _al = al[_idx_eul, :]
        _c_hb = c_hb[_idx_eul, :]
        _s_hb = s_hb[_idx_eul, :]
        _ep = ep[_idx_eul, :]
        
        _hh = h_lmp[i, :] * np.exp(1j*_mp*_ep)
        
        _A_p = np.zeros(n_pt, dtype=np.complex64)
        _A_c = np.zeros(n_pt, dtype=np.complex64)
        
        for j in range(-_lp, _lp+1, 1):
            _mm = j
            _Y = get_s_Ylm_sn2(th_s, phi_s, _lp, _mm)
            _dd_mp = get_Wigner_d_from_cs(_c_hb, _s_hb, _lp, _mm, _mp)
            _dd_n_mp = get_Wigner_d_from_cs(_c_hb, _s_hb, _lp, _mm, -_mp)
        
            exp_al_Y = np.exp(-1j*_mm*_al) * _Y
            
            A_p_mmp = _dd_mp * exp_al_Y + (-1)**_lp * _dd_n_mp * np.conjugate(exp_al_Y)                 
            A_c_mmp = _dd_mp * exp_al_Y - (-1)**_lp * _dd_n_mp * np.conjugate(exp_al_Y)
        
#             hp += 0.5 * _hh * A_p_mmp
#             hc += 0.5j * _hh * A_c_mmp
            _A_p += A_p_mmp
            _A_c += A_c_mmp
        
        hp += 0.5 * _hh * _A_p
        hc += 0.5j * _hh * _A_c
        
    return hp, hc   


@njit(parallel=True, fastmath=True, cache=True)
def get_h_iner_pol_from_h_prec_f_dmn_each_mode(al, c_hb, s_hb, ep, h_lmp, 
                                     ll_list_neg, mm_list_neg,
                                     th_s=0., phi_s=0.):
    """
    euler angles (al, beta, ep) have shapes (n_eul, n_freq)
    and the first index corresponds to [abs(mp) - min_abs_m]
    """    
    n_m_neg, n_pt = h_lmp.shape
    min_abs_m = int(np.min(np.abs(mm_list_neg)))
    
    h_pc_each_mode = np.zeros((2, n_m_neg, n_pt), dtype=np.complex64)
    
    for i in range(n_m_neg):
        _mp = mm_list_neg[i]
        _lp = ll_list_neg[i]

        _idx_eul = int(np.abs(_mp) - min_abs_m)
        _al = al[_idx_eul, :]
        _c_hb = c_hb[_idx_eul, :]
        _s_hb = s_hb[_idx_eul, :]
        _ep = ep[_idx_eul, :]
        
        _hh = h_lmp[i, :] * np.exp(1j*_mp*_ep)
        
        _A_p = np.zeros(n_pt, dtype=np.complex64)
        _A_c = np.zeros(n_pt, dtype=np.complex64)
        
        for j in range(-_lp, _lp+1, 1):
            _mm = j
            _Y = get_s_Ylm_sn2(th_s, phi_s, _lp, _mm)
            _dd_mp = get_Wigner_d_from_cs(_c_hb, _s_hb, _lp, _mm, _mp)
            _dd_n_mp = get_Wigner_d_from_cs(_c_hb, _s_hb, _lp, _mm, -_mp)
        
            exp_al_Y = np.exp(-1j*_mm*_al) * _Y
            
            A_p_mmp = _dd_mp * exp_al_Y + (-1)**_lp * _dd_n_mp * np.conjugate(exp_al_Y)                 
            A_c_mmp = _dd_mp * exp_al_Y - (-1)**_lp * _dd_n_mp * np.conjugate(exp_al_Y)
        
#             hp += 0.5 * _hh * A_p_mmp
#             hc += 0.5j * _hh * A_c_mmp
            _A_p += A_p_mmp
            _A_c += A_c_mmp
        
        h_pc_each_mode[0, i, :] = 0.5 * _hh * _A_p
        h_pc_each_mode[1, i, :] = 0.5j * _hh * _A_c
        
    return h_pc_each_mode   


@njit(parallel=True, fastmath=True, cache=True)
def get_h_iner_modes_from_h_prec_f_dmn(al, c_hb, s_hb, ep, h_lmp, 
                                     ll_list_neg, mm_list_neg):
    """
    euler angles (al, beta, ep) have shapes (n_eul, n_freq)
    and the first index corresponds to [abs(mp) - min_abs_m]
    
    Outputs: 
        h_lm: modes in the inertial frame with a shape (n_lm, n_freq)
              where n_lm = 5 * l2_flag + 7 * l3_flag + 9 * l4_flag
              with l2_flag = 1 if 2 in ll_list_neg, 0 otherwise, and similarly for other flags
        lm_list: shape (n_lm, 2)
                 l & m for each h_lm
    """    
    n_m_neg, n_pt = h_lmp.shape
    min_abs_m = int(np.min(np.abs(mm_list_neg)))
    
    l_flag = np.zeros(3, dtype=np.int8)
    n_per_l = np.array([5, 7, 9])
    if 2 in ll_list_neg:
        l_flag[0] = 1
    if 3 in ll_list_neg:
        l_flag[1] = 1
    if 4 in ll_list_neg:
        l_flag[2] = 1
        
    n_lm = np.sum(l_flag * n_per_l)
    h_lm = np.zeros((n_lm, n_pt), dtype=np.complex64)
    lm_list = np.zeros((n_lm, 2))
    
    # looping over co-precessing (lp, mp)
    for i in range(n_m_neg):
        _mp = mm_list_neg[i]
        _lp = ll_list_neg[i]

        _idx_eul = int(np.abs(_mp) - min_abs_m)
        _al = al[_idx_eul, :]
        _c_hb = c_hb[_idx_eul, :]
        _s_hb = s_hb[_idx_eul, :]
        _ep = ep[_idx_eul, :]
        
        _hh_p_exp_ep = h_lmp[i, :] * np.exp(1j*_mp*_ep)
        
        # the co-prec mode contributes to inertial modes with same l
        # find the starting index in h_lm of the relevant modes
        
        l_m_2 = int(_lp)-2
        idx_lm_0 = np.sum( (l_flag * n_per_l)[:l_m_2] )
        
        for j in range(-_lp, _lp+1, 1):
            _mm = j
            lm_list[idx_lm_0 + j + _lp, :] = np.array([_lp, _mm])
            
            _dd_mp = get_Wigner_d_from_cs(_c_hb, _s_hb, _lp, _mm, _mp)
            _h_lm = np.exp(-1j*_mm*_al) * _dd_mp * _hh_p_exp_ep
            
            h_lm[idx_lm_0 + j + _lp, :] += _h_lm
            
    return h_lm, lm_list




###############
# useful math #
###############

@njit(fastmath=True, cache=True)
def rot_pol(hp, hc, pol):
    _hp =  hp * np.cos(pol) + hc * np.sin(pol)
    _hc = -hp * np.sin(pol) + hc * np.cos(pol)
    return _hp, _hc


@njit(fastmath=True, cache=True)
def get_Wigner_d_from_cs_mp_pos(c_hb, s_hb, l, m, mp):
    l, m, mp = int(l), int(m), int(mp)
    
    c_hb_2 = c_hb * c_hb
    s_hb_2 = s_hb * s_hb
    
    if l==2:
        if mp==2:
            if m==2:
                dd = c_hb_2 * c_hb_2
            elif m==1:
                dd = 2 * c_hb_2 * c_hb * s_hb
            elif m==0:
                dd = 2.449489742783178 * c_hb_2 * s_hb_2
            elif m==-1:
                dd= 2 * c_hb * s_hb_2 * s_hb
            elif m==-2:
                dd = s_hb_2 * s_hb_2
                
        elif mp==1:
            if m==2:
                dd = -2 * c_hb_2 * c_hb * s_hb
            elif m==1:
                dd = c_hb_2 * (c_hb_2 - 3 * s_hb_2)
            elif m==0:
                dd = 2.449489742783178 * (c_hb_2 * c_hb * s_hb - c_hb * s_hb_2 * s_hb)
            elif m==-1:
                dd = s_hb_2 * (3 * c_hb_2 - s_hb_2)
            elif m==-2:
                dd = 2 * c_hb * s_hb_2 * s_hb
                
    elif l==3:
        if mp==3:
            if m==3:
                dd = c_hb_2 * c_hb_2 * c_hb_2
            elif m==2:
                dd = 2.449489742783178 * c_hb_2 * c_hb_2 * c_hb * s_hb
            elif m==1:
                dd = 3.872983346207417 * c_hb_2 * c_hb_2 * s_hb_2
            elif m==0:
                dd = 4.47213595499958 * c_hb_2 * c_hb * s_hb_2 * s_hb
            elif m==-1:
                dd = 3.872983346207417 * c_hb_2 * s_hb_2 * s_hb_2
            elif m==-2:
                dd = 2.449489742783178 * c_hb * s_hb_2 * s_hb_2 * s_hb
            elif m==-3:
                dd = s_hb_2 * s_hb_2 * s_hb_2
                
        elif mp==2:
            if m==3:
                dd = -2.449489742783178 * c_hb_2 * c_hb_2 * c_hb * s_hb
            elif m==2:
                dd = c_hb_2 * c_hb_2 * (c_hb_2 - 5 * s_hb_2)
            elif m==1:
                dd = 3.1622776601683795 * c_hb_2 * c_hb * (c_hb_2 * s_hb - 2 * s_hb_2 * s_hb)
            elif m==0:
                dd = 5.477225575051661 * c_hb_2 * s_hb_2 * (c_hb_2 - s_hb_2)
            elif m==-1:
                dd = 3.1622776601683795 * s_hb_2 * s_hb * (2 * c_hb_2 * c_hb - c_hb * s_hb_2)
            elif m==-2:
                dd = s_hb_2 * s_hb_2 * (5 * c_hb_2 - s_hb_2)
            elif m==-3:
                dd = 2.449489742783178 * c_hb * s_hb_2 * s_hb_2 * s_hb
                
        elif mp==1:
            if m==3:
                dd = 3.8729833462074166 * c_hb_2 * c_hb_2 * s_hb_2
            elif m==2:
                dd = 3.1622776601683795 * c_hb * s_hb * (-c_hb_2 * c_hb_2 + 2. * c_hb_2 * s_hb_2)
            elif m==1:
                dd = c_hb_2 * (c_hb_2*c_hb_2 - 8.*c_hb_2*s_hb_2 + 6.*s_hb_2*s_hb_2 )
            elif m==0:
                dd = 3.464101615137755 * c_hb * s_hb * (c_hb_2*c_hb_2 - 3.*c_hb_2*s_hb_2 + s_hb_2*s_hb_2)
            elif m==-1:
                dd = s_hb_2 * (6.*c_hb_2*c_hb_2 - 8*c_hb_2*s_hb_2 + s_hb_2*s_hb_2)
            elif m==-2:
                dd = 3.1622776601683795 * c_hb * s_hb * (2.*c_hb_2*s_hb_2 - s_hb_2*s_hb_2)
            elif m==-3:
                dd = 3.8729833462074166 * c_hb_2 * s_hb_2 * s_hb_2
                
        else:
            dd = np.zeros(len(c_hb)) # FIXME
                
    elif l==4:
        c_hb_4 = c_hb_2 * c_hb_2
        s_hb_4 = s_hb_2 * s_hb_2
        
        if mp==4:
            if m==4:
                dd = c_hb_4 * c_hb_4
            elif m==3:
                dd = 2.8284271247461903 * c_hb_4 * c_hb_2 * c_hb * s_hb
            elif m==2:
                dd = 5.291502622129181 * c_hb_4 * c_hb_2 * s_hb_2
            elif m==1:
                dd = 7.483314773547883 * c_hb_4 * c_hb * s_hb_2 * s_hb
            elif m==0:
                dd = 8.366600265340756 * c_hb_4 * s_hb_4
            elif m==-1:
                dd = 7.483314773547883 * c_hb_2 * c_hb * s_hb_4 * s_hb
            elif m==-2:
                dd = 5.291502622129181 * c_hb_2 * s_hb_4 * s_hb_2
            elif m==-3:
                dd = 2.8284271247461903 * c_hb * s_hb_4 * s_hb_2 * s_hb
            elif m==-4:
                dd = s_hb_4 * s_hb_4
                
        else:
            dd = np.zeros(len(c_hb)) # FIXME
            
    else:
        dd = np.zeros(len(c_hb))
                
    return dd

@njit(fastmath=True, cache=True)
def get_Wigner_d_from_cs(c_hb, s_hb, l, m, mp):
    if mp>0:
        dd = get_Wigner_d_from_cs_mp_pos(c_hb, s_hb, l, m, mp)
    else: 
        dd = get_Wigner_d_from_cs_mp_pos(c_hb, s_hb, l, -m, -mp)
        dd *= (-1)**(m + mp)
    return dd


@njit(fastmath=True, cache=True)
def get_s_Ylm_sn2(th_s, phi_s, l, m):
    """
    https://lscsoft.docs.ligo.org/lalsuite/lal/_spherical_harmonics_8c_source.html
    """
    c_th = np.cos(th_s)
    s_th = np.sin(th_s)
    
    if l==2:
        if m==-2:
            YY = 0.15769578262626002 * ( 1.0 - c_th )*( 1.0 - c_th )
        elif m==-1:
            YY = 0.31539156525252005 * s_th *( 1.0 - c_th )
        elif m==0:
            YY = 0.3862742020231896 * s_th * s_th
        elif m==1:
            YY = 0.31539156525252005 * s_th * ( 1.0 + c_th )
        elif m==2:
            YY = 0.15769578262626002 * ( 1.0 + c_th )*( 1.0 + c_th )
            
    elif l==3:
        if m==-3:
            YY = 1.828183197857863 * np.cos(th_s/2.) * pow(np.sin(th_s/2.), 5.0)
        elif m==-2:
            YY = 0.7463526651802308 * (2.0 + 3.0*c_th) * pow(np.sin(th_s/2.), 4.0)
        elif m==-1:
            YY = 0.07375544874083044 * (s_th + 4.0*np.sin(2.0*th_s) - 3.0*np.sin(3.0*th_s))
        elif m==0:
            YY =  1.0219854764332823 * c_th * pow(s_th, 2.0) 
        elif m==1:
            YY = -0.07375544874083044 * (s_th - 4.0*np.sin(2.0*th_s) - 3.0*np.sin(3.0*th_s))
        elif m==2:
            YY = 0.7463526651802308 * pow(np.cos(th_s/2.), 4.0) * (-2.0 + 3.0*c_th)
        elif m==3:
            YY = -1.828183197857863 * pow(np.cos(th_s/2.),5.0) * np.sin(th_s/2.0)
    
    elif l==4:
        if m==-4:
            YY = 4.478115991081385 * pow(np.cos(th_s/2.0),2.0) * pow(np.sin(th_s/2.0),6.0)
        elif m==-3:
            YY = 3.1665061842335644* np.cos(th_s/2.0)*(1.0 + 2.0*c_th) * pow(np.sin(th_s/2.0),5.0)
        elif m==-2:
            YY = (0.42314218766081724*(9.0 + 14.0*c_th + 7.0*np.cos(2.0*th_s)) * pow(np.sin(th_s/2.0),4.0))
        elif m==-1:
            YY = 0.03740083878763432*(3.0*s_th + 2.0*np.sin(2.0*th_s) + 7.0*np.sin(3.0*th_s) - 7.0*np.sin(4.0*th_s))
        elif m==0:
            YY = 0.1672616358893223*(5.0 + 7.0*np.cos(2.0*th_s)) * s_th * s_th
        elif m==1:
            YY = 0.03740083878763432*(3.0*s_th - 2.0*np.sin(2.0*th_s) + 7.0*np.sin(3.0*th_s) + 7.0*np.sin(4.0*th_s))
        elif m==2:
            YY = (0.42314218766081724*pow(np.cos(th_s/2.0),4.0)*(9.0 - 14.0*c_th + 7.0*np.cos(2.0*th_s)))
        elif m==3:
            YY = -3.1665061842335644*pow(np.cos(th_s/2.0),5.0)*(-1.0 + 2.0*c_th)*np.sin(th_s/2.0)
        elif m==4:
            YY = 4.478115991081385 * pow(np.cos(th_s/2.0),6.0)*pow(np.sin(th_s/2.0),2.0)
        
    else:
        YY = 0j
        
    YY *= np.exp(1j*m*phi_s)
        
    return YY
        