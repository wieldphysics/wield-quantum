#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import sys
import numpy as np
import math
import declarative

from numpy.polynomial.hermite import hermval
from scipy.special import eval_genlaguerre

from wavestate.utilities.np import matrix_stack, matrix_stack_id
from wavestate.utilities.mpl import (
    #generate_stacked_plot_ax,
    mplfigB, asavefig,
)
#asavefig.formats.png.use = True
from scipy.special import (
    eval_hermite,
    factorial,
)

from numpy.polynomial.hermite import hermval

c_m_s = 299792458

def hermite_exp(n, x):
    """
    From:
    https://www.numbercrunch.de/blog/2014/08/calculating-the-hermite-functions/
    """
    if n == 0:
        return np.ones_like(x)*np.pi**(-0.25)*np.exp(-x**2/2)
    if n == 1:
        return np.sqrt(2.)*x*np.exp(-x**2/2)*np.pi**(-0.25)
    h_i_2 = np.ones_like(x)*np.pi**(-0.25)
    h_i_1 = np.sqrt(2.)*x*np.pi**(-0.25)
    sum_log_scale = np.zeros_like(x)
    for i in range(2, n+1):
        h_i            = np.sqrt(2./i)*x*h_i_1-np.sqrt((i-1.)/i)*h_i_2
        h_i_2, h_i_1   = h_i_1, h_i
        log_scale      = np.log(abs(h_i) + 1e-14).round()
        scale          = np.exp(-log_scale)
        h_i            = h_i*scale
        h_i_1          = h_i_1*scale
        h_i_2          = h_i_2*scale
        sum_log_scale += log_scale
    return h_i*np.exp(-x**2/2+sum_log_scale)

def p_fock(n, p):
    if n < 100:
        fock = (
            (1/np.sqrt((2**n)*factorial(n)))*(np.pi**-0.25)
            * np.exp(-(p**2)/2)*eval_hermite(n, p)
        )
    else:
        fock = hermite_exp(n, p)
    return fock

def q_fock(n, q):
    '''Returns the Fock position wavefunctions defined over q space.
    
    Args:
        n (int): Fock index
        q (array): position values
    
    Returns:
        fock (array): nth Fock state wavefunction

    code from QuTiP
    '''
    if n < 100:
    #Employs the scipy.special eval_hermite function
        fock = (
            (1/np.sqrt((2**n)*factorial(n)))
            * (np.pi**-0.25)
            * np.exp(-(q**2)/2)
            * eval_hermite(n, q)
        )
    else:
        fock = (
            hermite_exp(n, q)
        )
    return fock

def basis_fock2q(n, q):
    M = np.empty((len(q), n))
    for k in range(0, n):
        M[:, k] = q_fock(n = k, q = q)
    return M

def raise_fock(n):
    return np.diagflat(np.arange(1, n)**0.5, -1)

def lower_fock(n):
    return np.diagflat(np.arange(1, n)**0.5, 1)


def linspace_clopen(half_width, N):
    return (np.arange(N) * (half_width / (N // 2)) - half_width)

def angle(a, shift = None, deg = False):
    if deg:
        mod = 360
    else:
        mod = np.pi * 2
    if shift is None:
        if deg:
            shift = -135
        else:
            shift = -3*np.pi/4
    return (np.angle(a, deg = deg) - shift) % mod + shift

def q2p(psi, q, extend = True):
    """
    FFT between q and p space. Assumes q is centered around 0 and evenly spaced
    """
    dq = q[1] - q[0]
    if extend:
        psi2 = np.concatenate([psi[len(psi)//2:], np.zeros(len(psi)), psi[:len(psi)//2]])
        psi_p = np.fft.fft(psi2) * dq / (np.pi * 2)**0.5
        psi_p = np.concatenate([psi_p[-len(psi)//2:], psi_p[:len(psi)//2]])
        #print(len(psi_p))
        p = linspace_clopen(np.pi/dq/2, len(q))
        return psi_p, p
    else:
        psi_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dq / (np.pi * 2)**0.5
        p = linspace_clopen(np.pi/dq, len(q))
        return psi_p, p


def q2wigner_fft(psi, q):
    """
    from qutip.wigner._wigner_fft
    """
    import scipy.linalg as la
    n = 2*len(psi)
    r1 = np.concatenate(((psi[::-1].conj()), np.zeros(n//2)), axis=0)
    r2 = np.concatenate((psi, np.zeros(n//2)), axis=0)
    w = la.toeplitz(np.zeros(n//2), r1) * np.flipud(la.toeplitz(np.zeros(n//2), r2))
    w = np.concatenate((w[:, n//2:n], w[:, 0:n//2]), axis=1)
    w = np.fft.fft(w)
    w = np.real(np.concatenate((w[:, 3*n//4:n+1], w[:, 0:n//4]), axis=1))
    p = np.arange(-n/4, n/4)*np.pi / (n*(q[1] - q[0]))
    p = q
    w = w / (p[1] - p[0]) / n
    return w, p


def rot45(rho, method = 'half'):
    n = rho.shape[-1]
    def update(k):
        k2 = k // 2
        x = np.diagonal(rho, -k)
        if k % 2 == 1:
            rho2[mid+k, k2:-k2-1] = np.diagonal(rho, k)/2
            if k2 == 0:
                rho2[mid+k, k2+1:] += np.diagonal(rho, k)/2
            else:
                rho2[mid+k, k2+1:-k2] += np.diagonal(rho, k)/2
        else:
            rho2[mid+k, k2:-k2] = np.diagonal(rho, k)
        x = np.diagonal(rho, k)
        if k % 2 == 1:
            rho2[mid-k, k2:-k2-1] = np.diagonal(rho, k)/2
            if k2 == 0:
                rho2[mid-k, k2+1:] += np.diagonal(rho, k)/2
            else:
                rho2[mid-k, k2+1:-k2] += np.diagonal(rho, k)/2
        else:
            rho2[mid-k, k2:-k2] = np.diagonal(rho, k)
    if method == 'half':
        rho2 = np.zeros((n, n), dtype = complex)
        mid = n//2
        rho2[mid, :] = np.diagonal(rho)
        for k in range(1, n//2):
            update(k)
    elif method == 'full':
        n2 = 2*n
        rho2 = np.zeros((n2, n), dtype = complex)
        mid = n
        rho2[mid, :] = np.diagonal(rho)
        for k in range(1, n):
            update(k)
    elif method == 'hermitian':
        n2 = 2*n
        rho2 = np.zeros((n, n), dtype = complex)
        rho2[0, :] = np.diagonal(rho)
        for k in range(1, n):
            k2 = k // 2
            x = np.diagonal(rho, k)
            if k % 2 == 1:
                rho2[k, k2:-k2-1] = np.diagonal(rho, k)/2
                if k2 == 0:
                    rho2[k, k2+1:] += np.diagonal(rho, k)/2
                else:
                    rho2[k, k2+1:-k2] += np.diagonal(rho, k)/2
            else:
                rho2[k, k2:-k2] = np.diagonal(rho, k)
    else:
        raise RuntimeError('Unrecognized Method')
    return rho2

def psiq2wigner_fft(psi, q, method = 'full'):
    rho = psi.reshape(-1, 1) * psi.conjugate().reshape(1, -1)
    return rhoq2wigner_fft(rho, q, method = method)

def rhoq2wigner_fft(rho, q, method = 'hermitian'):
    """
    """
    assert(rho.shape[0] == rho.shape[1])
    n = rho.shape[0]
    n2 = 2*n
    if method == 'full':
        w = rot45(rho, method = 'full')
        w = np.fft.ifftshift(w, axes = 0)
        w = np.fft.fft(w, axis = 0)
        w = np.real(np.concatenate((w[3*n2//4:, :], w[:n2//4, :]), axis=0))
        p = np.arange(-n/2, n/2)*np.pi / (n*(q[1] - q[0]))
    elif method == 'half':
        w = rot45(rho, method = 'half')
        w = np.fft.ifftshift(w, axes = 0)
        w = np.fft.fft(w, axis = 0)
        w = np.fft.fftshift(w, axes = 0)
        w = np.real(w)
        p = np.arange(-n/2, n/2)*2*np.pi / (n*(q[1] - q[0]))
    elif method == 'hermitian':
        w = rot45(rho, method = 'hermitian')
        w = np.fft.hfft(w, axis=0)
        w = np.concatenate((w[3*n2//4:, :], w[:n2//4, :]), axis=0)
        p = np.arange(-n/2, n/2)*np.pi / (n*(q[1] - q[0]))

    return w, p


def gkp(q, D, mu=0, s=None):
    '''Returns a discretized q space and a normalized logical (0 or 1) 
    GKP state. Default is to return logical 0 (mu=0).

    Args:
        q (float array): wavefunction domain
        D (float): width of the Gaussian envelope and squeezed states 
            in the superposition
        s (int): maximum number of tines in the comb
        mu (0 or 1): logical value for the GKP state

    Returns:
        q (array): position values
        gkp (array): array of GKP wavefunction values.


    From eq (35) in PhysRevA.64.012310
    '''
    if s is None:
        s = 2 * (int(q[-1]) + 1)

    if mu == 0:
        s_arr = np.arange(-s//2, s//2+1)
    else:
        s_arr = np.arange(-s//2, s//2+2)

    # Initiating empty arrays to be filled in
    gkp = np.zeros(len(q))

    # Looping over all peaks in the superposition
    for s_val in s_arr:
        #gkp += (
        #    np.exp(-((D**-2)*np.pi*(2*s_val+mu)**2)*2.0)
        #    * np.exp(-(q-(2*s_val+mu)*np.sqrt(np.pi))**2/2 * (D**2))
        #    / (np.pi * D**2)**-0.25
        #)
        gkp += (
            np.exp(-2 * np.pi * D**2 * (s_val + mu/2)**2)
            * np.exp(-(q - (2 * s_val + mu) * np.sqrt(np.pi))**2 / (D**2) / 2)
            * (4 / np.pi)**0.25
        )

    # Normalization constants
    # GKP gives a good approximation of sqrt(2*Delta) if Delta/alpha is small,
    # but we can find them numerically to get a better approximation
    #dq = q[1] - q[0]

    #no need to normalize
    #Norm = 1#/np.sqrt(dq * np.sum(np.absolute(gkp)**2))
    # Normalizing
    #gkp = Norm*gkp

    return gkp
