# coding: utf-8
"""
"""
import sys
import numpy as np
import math
import declarative

from numpy.polynomial.hermite import hermval
from scipy.special import eval_genlaguerre

from transient.matrix import matrix_stack, matrix_stack_id
from transient.utilities.mpl import (
    #generate_stacked_plot_ax,
    mplfigB, asavefig,
)
#asavefig.formats.png.use = True

from transient.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot
)

from scipy.linalg import expm
from transient.quantum import fock

c_m_s = 299792458


def T_fock(tpath_join, pprint, plot):
    N = 100
    q = fock.linspace_clopen(N, N**2 / 2)
    psi0 = fock.q_fock(n = 0, q = q)
    psi1 = fock.q_fock(n = 1, q = q)
    psi2 = fock.q_fock(n = 2, q = q)
    psi3 = fock.q_fock(n = 8, q = q)
    dq = q[1] - q[0]

    #pprint(len(q), len(q)//2, q[len(q)//2-1], q[len(q)//2], q[len(q)//2+1])
    #pprint(q[0], q[-1])

    axB = mplfigB(Nrows = 3)

    #this block to test the exact values needed for q, now in linspace_clopen
    #psi0 = np.fft.ifftshift(psi0)
    #psi1 = np.fft.ifftshift(psi1)
    #psi2 = np.fft.ifftshift(psi2)
    #psi3 = np.fft.ifftshift(psi3)
    #pprint(psi0[0], psi0[-1])

    #axB_ = mplfigB()
    #axB_.ax0.plot(psi0[:len(psi0)//2-1:-1] - psi0[:len(psi0)//2])
    #axB_.save(tpath_join('test'))
    axB.ax0.plot(q, dq*np.cumsum(abs(psi0)**2))
    axB.ax0.plot(q, dq*np.cumsum(abs(psi1)**2))
    axB.ax0.plot(q, dq*np.cumsum(abs(psi2)**2))
    axB.ax0.plot(q, dq*np.cumsum(abs(psi3)**2))
    axB.ax0.set_xlim(-10, 10)

    axB.ax1.plot(q, abs(psi0))
    axB.ax1.plot(q, abs(psi1))
    axB.ax1.plot(q, abs(psi2))
    axB.ax1.plot(q, abs(psi3))
    axB.ax1.set_xlim(-10, 10)

    axB.ax2.plot(q, fock.angle(psi0))
    axB.ax2.plot(q, fock.angle(psi1))
    axB.ax2.plot(q, fock.angle(psi2))
    axB.ax2.plot(q, fock.angle(psi3))
    axB.ax2.set_xlim(-10, 10)

    axB.save(tpath_join('fock_psi'))

    psi0_p, p = fock.q2p(psi0, q)
    psi1_p, p = fock.q2p(psi1, q)
    psi2_p, p = fock.q2p(psi2, q)
    psi3_p, p = fock.q2p(psi3, q)

    dp = p[1] - p[0]
    axB = mplfigB(Nrows = 3)
    axB.ax0.plot(p, dp * np.cumsum(abs(psi0_p)**2))
    axB.ax0.plot(p, dp * np.cumsum(abs(psi1_p)**2))
    axB.ax0.plot(p, dp * np.cumsum(abs(psi2_p)**2))
    axB.ax0.plot(p, dp * np.cumsum(abs(psi3_p)**2))
    axB.ax0.set_xlim(-10, 10)

    axB.ax1.plot(p, (abs(psi0_p)))
    axB.ax1.plot(p, (abs(psi1_p)))
    axB.ax1.plot(p, (abs(psi2_p)))
    axB.ax1.plot(p, (abs(psi3_p)))
    axB.ax1.set_xlim(-10, 10)
    
    axB.ax2.plot(p, fock.angle(psi0_p))
    axB.ax2.plot(p, fock.angle(psi1_p))
    axB.ax2.plot(p, fock.angle(psi2_p))
    axB.ax2.plot(p, fock.angle(psi3_p))
    axB.ax2.set_xlim(-10, 10)
    axB.save(tpath_join('fock_psi_p'))
    return


def T_wigner(tpath_join, pprint, plot):
    q = fock.linspace_clopen(30, 1024)
    psi0 = fock.q_fock(n = 2, q = q)
    for method in ['full', 'half', 'hermitian']:
        axB = mplfigB()
        w, p = fock.psiq2wigner_fft(psi0, q, method = method)
        axB.ax0.set_aspect(1)
        minmax = np.max(abs(w))
        axB.ax0.imshow(
            w,
            extent = (q[0], q[-1], p[0], p[-1]),
            cmap = 'PiYG',
            vmin = -minmax,
            vmax = minmax,
            interpolation = 'nearest',
        )
        pprint(minmax)
        axB.ax0.grid(b=False)
        #axB.ax0.set_xlim(-10, 10)
        #axB.ax0.set_ylim(-10, 10)
        axB.save(tpath_join('wigner_{}'.format(method)))
    return


def T_fockdisp(tpath_join, pprint, plot):
    n = 50
    q = fock.linspace_clopen(10, 2048)
    dq = q[1] - q[0]
    psi = np.zeros(n, dtype = complex)
    #vacuum state
    psi[0] = 1
    qM = fock.basis_fock2q(n = n, q = q)

    def plot_psi(psi, name = None, line = None):
        psi_q = qM @ psi
        axB = mplfigB(Nrows = 3)
        axB.ax0.plot(q, dq*np.cumsum(abs(psi_q)**2))
        axB.ax1.plot(q, abs(psi_q))
        axB.ax2.plot(q, fock.angle(psi_q))
        if line is not None:
            axB.ax0.axvline(line)
        if name is not None:
            axB.save(tpath_join(name))
        
    plot_psi(psi, 'vac')

    aM = fock.lower_fock(n = n)
    plot_psi(adj(aM) @ psi, 'raised')

    pprint("Lowering Operator")
    pprint(aM)
    #pprint(adj(aM))
    pprint("Number Operator")
    pprint(adj(aM) @ aM)

    #def expm(A):
    #    sq = A @ A
    #    return np.eye(A.shape[0]) + A + sq / 2 + sq @ A / 6 + sq @ sq / 24

    alpha = np.array(2)
    disp = expm(alpha.conj() * adj(aM) - alpha * aM)
    pprint(disp @ psi)

    plot_psi(disp @ psi, 'disp', line = abs(alpha) * np.pi / 2)

    z = np.array(.5)
    aM2 = aM @ aM
    pprint(aM2)
    sqz = expm(z.conj() * aM2 - z * adj(aM2))
    pprint(sqz @ psi)

    plot_psi(sqz @ psi, 'sqz')
    return

def T_focksqz(tpath_join, pprint, plot):
    n = 100
    q = fock.linspace_clopen(50, 2048)
    rho = np.zeros((n, n), dtype = complex)
    #vacuum state
    rho[0, 0] = 1
    qM = fock.basis_fock2q(n = n, q = q)

    def plot_rho_wigner(rho, name = None):
        rho_q = qM @ rho @ adj(qM)
        w, p = fock.rhoq2wigner_fft(rho = rho_q, q = q)
        axB = mplfigB()
        axB.ax0.set_aspect(1)
        minmax = np.max(abs(w))
        axB.ax0.imshow(
            w,
            extent = (q[0], q[-1], p[0], p[-1]),
            cmap = 'PiYG',
            vmin = -minmax,
            vmax = minmax,
            interpolation = 'nearest',
        )
        pprint(minmax)
        axB.ax0.grid(b=False)
        axB.ax0.set_xlim(-10, 10)
        axB.ax0.set_ylim(-10, 10)
        axB.ax0.axvline(1)
        if name is not None:
            axB.save(tpath_join(name))
        
    plot_rho_wigner(rho, 'wigner_vac')

    aM = fock.lower_fock(n = n)
    plot_rho_wigner(adj(aM) @ rho @ aM, 'wigner_raised')

    pprint(aM)
    alpha = np.array(0.5)
    disp = expm(alpha.conj() * adj(aM) - alpha * aM)
    disp = disp @ disp

    plot_rho_wigner(disp @ rho @ adj(disp), 'wigner_disp')
    return

def adj(M):
    return M.T.conj()


