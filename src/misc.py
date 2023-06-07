import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import scipy.special as special
import h5py as h5
from numba import jit, njit, prange

from .LAL_constants import *
from . import precLib as pLib
from . import waveLib as wLib

#######################
# physical quantities #
####################### 

@njit
def get_Mc(M1, M2):
    Mc=(M1*M2)**(3./5.)/(M1+M2)**(1./5.)
    return Mc

@njit
def get_t_merger(f_gw, M1, M2):
    Mc=get_Mc(M1, M2)
    t_m=(5./(256.*np.pi**(8./3.))) \
        * (c**3./(G*Mc))**(5./3.)*f_gw**(-8./3.)
    return t_m

@njit
def est_f_lower_from_t_m(t_m, M1, M2):
    Mc = get_Mc(M1, M2)
    f_low = (5./256.)**(0.375)/np.pi * (c**3./G/Mc)**0.625 * t_m**(-0.375)
    return f_low


#####################
# signal processing #
#####################

@njit
def get_delta_t_delta_f(f_lower, M1, M2):
    Mt = M1+M2
    t_Mt = G*Mt/c**3.
    
    # determine dt based on ringdown freq
    f_max_est = 0.3/t_Mt 
    f_max_est = 2.**np.ceil(np.log2(f_max_est))

    delta_t = 1./2/f_max_est
    
    # determine df based on the signal duration
    t_dur_est = get_t_merger(f_lower, M1, M2)
    n_td_est = t_dur_est / delta_t
    n_td_est = 2.**np.ceil(np.log2(n_td_est))
    t_dur_est = n_td_est * delta_t

    delta_f = 1./t_dur_est
    return delta_t, delta_f


def get_fd_from_td(h_td, delta_t, delta_f, 
                   t0=0):
    n_td = len(h_td)
    
    # FIXME
    # just an arbitrary choice of window
    win = sig.tukey(n_td, 0.25)
    # win = sig.hamming(n_td)
    win = np.hstack((win[:n_td//2], np.ones_like(win[n_td//2:])))
    
    win_end = sig.hann(16)
    win[-8:]=win_end[-8:]
    
    n_td_tot = int(1./delta_f * 1./delta_t) #2**int(np.ceil(np.log2(n_td)))
    n_td_pad = n_td_tot - n_td
    n_td_pad_aft = n_td_pad //2
    n_td_pad_b4 = n_td_pad - n_td_pad_aft
    h_td_pad = np.pad(np.real(h_td)*win, (n_td_pad_b4, n_td_pad_aft))

    t0 -= delta_t * n_td_pad_b4
    
    h_fd = np.fft.rfft(h_td_pad) * delta_t
    f_fd = np.fft.rfftfreq(n_td_tot, delta_t)
    
    h_fd *= np.exp(-1j*2.*np.pi*t0*f_fd) 
    return f_fd, h_fd

def read_mag(freq, fileName, fill_value='extrapolate'):
    f_tf, mag_tf=np.loadtxt(fileName, unpack=True)
    
    idx=np.where(f_tf>0)
    mag_tf=mag_tf[idx]
    f_tf=f_tf[idx]

    mag_func=interp.interp1d(np.log(f_tf), np.log(mag_tf), \
                    kind='linear', bounds_error=False, fill_value=fill_value)
    mag_out=np.exp(mag_func(np.log(freq)))
    return mag_out


####################
# match b/t models #
####################

@njit(cache=True)
def inner_prod(freq, h1, h2, psd, 
               f_min=-np.infty, f_max=np.infty):
    match = np.real(h1 * np.conj(h2))
    match_integrand = 4.*match/psd    
    idx = (freq>f_min) & (freq<f_max)    
    match = np.trapz(match_integrand[idx], freq[idx])
    return match

@njit(cache=True)    
def inner_prod_complex(freq, h1, h2, psd, 
               f_min=-np.infty, f_max=np.infty):
    match = h1 * np.conj(h2)
    match_integrand = 4.*match/psd    
    idx = (freq>f_min) & (freq<f_max)
    match = np.trapz(match_integrand[idx], freq[idx])
    return match

@njit(cache=True)
def inner_prod_integrand(freq, h1, h2, psd):
    match = h1 * np.conj(h2)
    match_integrand = 4.*match/psd 
    return match_integrand

def ang_diff_lin_free(freq, h1, h2, psd, 
                      f_min=-np.infty, f_max=np.infty, 
                      ww_sqrt_f_cut=3):
    _h2 = get_h2_lin_free(freq, h1, h2, psd, 
                    f_min=f_min, f_max=f_max, 
                    ww_sqrt_f_cut=ww_sqrt_f_cut)
    ang_diff = np.angle(h1 * _h2.conj())
    return ang_diff

def get_h2_lin_free(freq, h1, h2, psd, 
                    f_min=-np.infty, f_max=np.infty, 
                    ww_sqrt_f_cut=3):
    
    ww = np.sqrt(np.abs(h1*h2/psd) )
    ww_sqrt_f = np.sqrt(freq) * ww
        
    idx = (freq>f_min) & (freq<f_max)
    ww_sqrt_f_max = np.max(ww_sqrt_f[idx])
    
    idx_w = idx & (ww_sqrt_f > ww_sqrt_f_max/ww_sqrt_f_cut)
    
#     print(f_min, f_max, ww_sqrt_f_max)
#     print(freq[idx])
#     print(freq[idx_w])
    
    # initial fit
    ang_diff = np.unwrap( np.angle(h1 * h2.conj())[idx_w] )
    
    _2pi_tc_phic = np.polyfit(freq[idx_w], ang_diff, deg=1, w=ww[idx_w])
    
    _2pi_tc = _2pi_tc_phic[0]
    _phic = _2pi_tc_phic[1]
    _h2 = h2.copy()
    _h2 *= np.exp(1j*(freq*_2pi_tc + _phic))
    
    
    #
    match_0 = inner_prod(freq, h1, _h2, psd, f_min, f_max)
    ang_diff = np.unwrap( np.angle(h1 * _h2.conj())[idx] )
    _2pi_tc_phic = np.polyfit(freq[idx], ang_diff, deg=1, w=ww[idx])    
    _2pi_tc = _2pi_tc_phic[0]
    _phic = _2pi_tc_phic[1]
    __h2 = _h2 * np.exp(1j*(freq*_2pi_tc + _phic))     
    match_1 = inner_prod(freq, h1, __h2, psd, f_min, f_max)
        
    if (np.max(ang_diff) - np.min(ang_diff) < 2.*np.pi) \
        and (match_1 >= match_0):
        # unwrapping was succsessful
        _h2 = __h2
            
    else:
        # unwrapping was unsuccsessful
        # need to numerically remove tc & phic
        _2pi_tc_phic = opt.minimize(opt_2pi_tc_phic_func, (0., 0.), 
                                    args=(freq, h1, _h2, psd, f_min, f_max), 
                                    method='Nelder-Mead')    
#         _2pi_tc_phic = opt.dual_annealing(opt_2pi_tc_phic_func, ((-0.03, 0.03), (-0.3*np.pi, 0.3*np.pi)), 
#                                          args=(freq, h1, _h2, psd, f_min, f_max), x0=(0., 0.))

#         if not _2pi_tc_phic.success:
#             _2pi_tc_phic = opt.dual_annealing(opt_2pi_tc_phic_func, ((-0.3, 0.3), (-3.*np.pi, 3.*np.pi)), 
#                                          args=(freq, h1, _h2, psd, f_min, f_max), x0=(0., 0.))
        _2pi_tc_phic = _2pi_tc_phic.x
        
        _2pi_tc = _2pi_tc_phic[0]
        _phic = _2pi_tc_phic[1]
        _h2 *= np.exp(1j*(freq*_2pi_tc + _phic))     
    return _h2

@njit
def opt_2pi_tc_phic_func(xx, freq, h1, h2, psd, 
                   f_min=-np.infty, f_max=np.infty):
    _2pi_tc, _phic = xx
    _h2 = h2 * np.exp(1j*(freq*_2pi_tc+_phic))
    match_n = -inner_prod(freq, h1, _h2, psd, f_min, f_max)
    return match_n
    

def pol_max_match(freq, ss, h0_pc, psd, 
                  f_min=-np.infty, f_max=np.infty, 
                  n_iter=2, kappa_0 = 0, 
                  ww_sqrt_f_cut=3):
    """
    Harry+ 2016, PRD 94, 024012 (2016)
    
    approximate time alignment required
    """
    h0_p, h0_c = h0_pc[0, :], h0_pc[1, :]
    
    # norm
    h0_p_norm = np.sqrt(inner_prod(freq, h0_p, h0_p, psd, f_min, f_max))
    h0_c_norm = np.sqrt(inner_prod(freq, h0_c, h0_c, psd, f_min, f_max))
    
    pc_ratio = h0_p_norm / h0_c_norm
    
    _h0_p = h0_p/h0_p_norm
    _h0_c = h0_c/h0_c_norm
    
    # remove time shift first
    h0 = _h0_p * np.cos(kappa_0) + _h0_c * np.sin(kappa_0)
    h0 /= np.sqrt(inner_prod(freq, h0, h0, psd, f_min, f_max))
    
    _h0 = get_h2_lin_free(freq, ss, h0, psd, 
                         f_min=f_min, f_max=f_max, 
                         ww_sqrt_f_cut=ww_sqrt_f_cut)
    ang_lin = np.angle(_h0 * h0.conj())
    
    _h0_p *= np.exp(1j*ang_lin)
    _h0_c *= np.exp(1j*ang_lin)
    
    match_0 = inner_prod(freq, ss, _h0, psd, f_min, f_max)
    
    cnt = 0
    while cnt<n_iter:
        # since time shift and polarizations are separately optimized
        # iterate over the optimization process
        
        # opt match over polarization
        rho_p = inner_prod_complex(freq, ss, _h0_p, psd, f_min, f_max)
        rho_c = inner_prod_complex(freq, ss, _h0_c, psd, f_min, f_max)
        
        rho_p_sq = np.abs(rho_p)**2
        rho_c_sq = np.abs(rho_c)**2
        
        gamma = np.real(rho_p * np.conj(rho_c))
        I_pc = np.real(inner_prod_complex(freq, _h0_p, _h0_c, psd, f_min, f_max))
        
        I_p_term = I_pc * rho_p_sq - gamma
        I_c_term = I_pc * rho_c_sq - gamma
        sqrt_term = np.sqrt((rho_p_sq - rho_c_sq)**2. + 4*I_p_term*I_c_term)
        
        u_num = - (rho_p_sq - rho_c_sq) - sqrt_term # - here corresponds to + in eq. 26
        u_den = 2 * I_p_term
    
        # optimal polarization
        kappa = np.arctan2(u_den*pc_ratio, u_num)
#         print('kappa/pi', kappa/np.pi)
    
        match = 0.5 * (rho_p_sq - 2.*gamma*I_pc + rho_c_sq + sqrt_term) \
            / (1.-I_pc**2)
        match = np.sqrt(match)
#         print('match:', match)
    
        # now form template based on the opt pol
        # and re-do time & phase shift using the new waveform
        h0 = _h0_p * np.cos(kappa) + _h0_c * np.sin(kappa)
        h0 /= np.sqrt(inner_prod(freq, h0, h0, psd, f_min, f_max))
    
        _h0 = get_h2_lin_free(freq, ss, h0, psd, 
                             f_min=f_min, f_max=f_max, 
                             ww_sqrt_f_cut=ww_sqrt_f_cut)
        ang_lin = np.angle(_h0 * h0.conj())
    
        _h0_p *= np.exp(1j*ang_lin)
        _h0_c *= np.exp(1j*ang_lin)      
        
        cnt+=1
    return match, kappa, match_0


def linfree_phi_ref_phi_az_func(xx, ss, psd, kwargs_hh, 
                              kappa_0=0., f_min=-np.infty, f_max=np.infty):
    
    _dphi_ref, _phi_az = xx
#     _dphi_ref, = xx
    
    freq = kwargs_hh['freqs']
    
    phi_ref0 = kwargs_hh['phi_ref']
    phi_ref = phi_ref0 + _dphi_ref
    
    c_az, s_az = np.cos(_phi_az), np.sin(_phi_az)
    
    s1x, s1y = kwargs_hh['spin1x'], kwargs_hh['spin1y']
    s2x, s2y = kwargs_hh['spin2x'], kwargs_hh['spin2y']
    
    _s1x =  s1x * c_az + s1y * s_az
    _s1y = -s1x * s_az + s1y * c_az
    _s2x =  s2x * c_az + s2y * s_az
    _s2y = -s2x * s_az + s2y * c_az    
    
    # updating phi_ref and in-plane spins
    _kwargs = kwargs_hh.copy()
    _kwargs['phi_ref'] = phi_ref
    _kwargs['spin1x'], _kwargs['spin1y'] = _s1x, _s1y
    _kwargs['spin2x'], _kwargs['spin2y'] = _s2x, _s2y
    
    _hp, _hc = wLib.get_hp_hc_f_sequence(**_kwargs)
    
    _h_pc = np.vstack((_hp, _hc))
    
    match, __, __ = pol_max_match(freq, ss, _h_pc, psd, 
                  f_min=f_min, f_max=f_max, 
                  n_iter=1, kappa_0 = kappa_0)
        
#     print(xx, 1-match)
    return 100.*(1.-match)

def opt_phi_ref_phi_az(ss, psd, kwargs_hh, 
                      kappa_0=0., f_min=-np.infty, f_max=np.infty, 
                      dual_annealing=False):
    if dual_annealing:    
        ret = opt.dual_annealing(linfree_phi_ref_phi_az_func, 
                                ((-np.pi, np.pi), (-np.pi, np.pi)), 
                                args=(ss, psd, kwargs_hh, kappa_0, f_min, f_max), x0=(0., 0.), 
                                maxfun=1e5, maxiter=250)
        xx = ret.x
    else:
        ret = opt.minimize(linfree_phi_ref_phi_az_func, 
                           x0=(0., 0.), args=(ss, psd, kwargs_hh, kappa_0, f_min, f_max),
                           bounds=((-np.pi, np.pi), (-np.pi, np.pi)), method='Nelder-Mead',
                          )    
        xx = ret.x
        mm = ret.fun
        x0_list = [(0.45*np.pi, -0.1), (-0.45*np.pi, 0.1)]
        for i in range(len(x0_list)):
            _x0 = x0_list[i]
            
            _ret = opt.minimize(linfree_phi_ref_phi_az_func, 
                           x0=_x0, args=(ss, psd, kwargs_hh, kappa_0, f_min, f_max),
                           bounds=((-np.pi, np.pi), (-np.pi, np.pi)), method='Nelder-Mead',
                          )
            if _ret.fun < mm:
                mm = _ret.fun
                xx = _ret.x            
    _dphi_ref, _phi_az = xx
    
    freq = kwargs_hh['freqs']
    
    phi_ref0 = kwargs_hh['phi_ref']
    phi_ref = phi_ref0 + _dphi_ref
    
    c_az, s_az = np.cos(_phi_az), np.sin(_phi_az)
    
    s1x, s1y = kwargs_hh['spin1x'], kwargs_hh['spin1y']
    s2x, s2y = kwargs_hh['spin2x'], kwargs_hh['spin2y']
    
    _s1x =  s1x * c_az + s1y * s_az
    _s1y = -s1x * s_az + s1y * c_az
    _s2x =  s2x * c_az + s2y * s_az
    _s2y = -s2x * s_az + s2y * c_az    
    
    # updating phi_ref and in-plane spins
    _kwargs = kwargs_hh.copy()
    _kwargs['phi_ref'] = phi_ref
    _kwargs['spin1x'], _kwargs['spin1y'] = _s1x, _s1y
    _kwargs['spin2x'], _kwargs['spin2y'] = _s2x, _s2y
    
    _hp, _hc = wLib.get_hp_hc_f_sequence(**_kwargs)
    
    _h_pc = np.vstack((_hp, _hc))
    
    match, kappa, __ = pol_max_match(freq, ss, _h_pc, psd, 
                  f_min=f_min, f_max=f_max, 
                  n_iter=1, kappa_0 = kappa_0)
    
    _h0 = _hp * np.cos(kappa) + _hc * np.sin(kappa)
    _h0 = get_h2_lin_free(freq, ss, _h0, psd, f_min=f_min, f_max=f_max)
    
    return match, _h0, xx
