#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np

from wavestate.utilities.mpl import (  # noqa
    #generate_stacked_plot_ax,
    mplfigB, asavefig,
)

from wavestate.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot
)

from wavestate.quantum import qop, qop_fock, qop_qubit, fock  # noqa


def T_sqz_wigner_loss(tpath_join, pprint, plot):
    L = 0.5
    bQ = qop_fock.basis_Q(N = 2*1024)
    bF = qop_fock.basis_Fock(N = 30)
    F2Q = qop_fock.basis_change('a', bC=bF, bR=bQ)

    bFx = qop_fock.basis_Fock(N = 2)
    psi_a = qop_fock.psi_vacuum('a', bN=bF)
    psi_b = qop_fock.psi_vacuum('b', bN=bF)
    psi = psi_a @ psi_b

    op_a = qop_fock.op_ladder_dn('a', bN=bF)
    op_b = qop_fock.op_ladder_dn('b', bN=bF)

    op_bs = (.5 * (op_a @ op_b.A - op_a.A @ op_b)).expm()
    pprint('op_bs', op_bs.mat)
    pprint('op_bs_index', op_bs.index_order)
    op_bs.segments_set([('b',), ('a',)])
    pprint('op_bs_index', op_bs.index_order)

    dbs = [-3, -6, -9]
    axBall = mplfigB(Nrows = len(dbs), Ncols = 2)
    for idx, db in enumerate(dbs):
        z = np.log(10**(db/10.)) / 2

        sqz_gen = z * op_a @ op_a
        sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()
        psi_sqz = op_bs @ sqz @ psi

        rho_sqz = psi_sqz @ psi_sqz.A
        rho_sqz = rho_sqz.trace_other('a')

        #pprint(len(rho_sqz.space_basis), rho_sqz.space_basis[0].bC.basis, rho_sqz.space_basis[0].bR.basis)
        rho_sqz = F2Q @ rho_sqz @ F2Q.A
        #pprint(len(rho_sqz.space_basis), rho_sqz.space_basis[0].bC.basis, rho_sqz.space_basis[0].bR.basis)

        axB = mplfigB(Ncols = 2)
        plot_wigner(axB.ax0_0, 'a', rho = rho_sqz, bQ=bQ, lims=10)
        #axB.ax0_1.stem(abs(psi_sqz.mat[:, 0])**2, markerfmt='C0,')
        axB.ax0_1.set_xscale('log')
        axB.save(tpath_join('wigner_sqz_{:.1f}db'.format(-db).replace('.', 'p')))
        plot_wigner(axBall['ax{}_0'.format(idx)], 'a', rho = rho_sqz, bQ=bQ, lims=10)
        axall = axBall['ax{}_1'.format(idx)]
        #axall.stem(abs(psi_sqz.mat[:, 0])**2, markerfmt='C0,')
        axall.set_xscale('log')
    axBall.save(tpath_join('wigner_sqz'.format(-db)))
    return


def T_gkp_wigner_loss(tpath_join, pprint, plot):
    bQ = qop_fock.basis_Q(N = 2*1024)
    bF = qop_fock.basis_Fock(N = 300)
    F2Q = qop_fock.basis_change('field', bC=bF, bR=bQ)

    iDs = [2, 4, 5, 8]
    axBall = mplfigB(Nrows = len(iDs), Ncols = 2)
    for idx, iD in enumerate(iDs):
        psi_gkp = F2Q.A @ qop_fock.psi_gkp('field', bN=bQ, D=1/iD, mu=0)

        pprint('iD', iD)
        qop_fock.op_qp('field', bN = bF)

        axB = mplfigB(Ncols = 2)
        plot_wigner(axB.ax0_0, 'field', psi = F2Q @ psi_gkp, bQ=bQ, lims=10)
        axB.ax0_1.stem(abs(psi_gkp.mat[:, 0])**2, markerfmt='C0,')
        axB.ax0_1.set_xscale('log')
        axB.save(tpath_join('wigner_gkp_{:.0f}'.format(iD).replace('.', 'p')))
        plot_wigner(axBall['ax{}_0'.format(idx)], 'field', psi = F2Q @ psi_gkp, bQ=bQ, lims=10)
        axall = axBall['ax{}_1'.format(idx)]
        axall.stem(abs(psi_gkp.mat[:, 0])**2, markerfmt='C0,')
        axall.set_xscale('log')
    axBall.save(tpath_join('wigner_gkp'))
    return


def plot_wigner(ax, *args, **kwargs):
    lims = kwargs.pop('lims', None)
    w, q, p = qop_fock.qop2wigner(*args, **kwargs)
    ax.set_aspect(1)
    minmax = np.max(abs(w))
    ax.imshow(
        w,
        extent = (q[0], q[-1], p[0], p[-1]),
        cmap = 'PiYG',
        vmin = -minmax,
        vmax = minmax,
        interpolation = 'nearest',
    )
    ax.grid(b=False)
    if lims is not None:
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
