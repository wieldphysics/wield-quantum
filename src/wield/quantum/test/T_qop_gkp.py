#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np

from wield.quantum import qop, qop_fock, qop_qubit, fock  # noqa

from wield.utilities.mpl import (  # noqa
    # generate_stacked_plot_ax,
    mplfigB,
    asavefig,
)

from wield.pytest.fixtures import (  # noqa: F401
    tjoin,
    dprint,
    plot
)


def T_sqz_wigner_loss(plot):
    L = 0.5
    bQ = qop_fock.basis_Q(N=2 * 1024)
    bF = qop_fock.basis_Fock(N=10)
    F2Q = qop_fock.basis_change("a", bC=bF, bR=bQ)

    bFx = qop_fock.basis_Fock(N=2)
    psi = qop_fock.psi_vacuum("a", bN=bF)

    op_a = qop_fock.op_ladder_dn("a", bN=bF)
    op_b = qop_fock.op_ladder_dn("b", bN=bF)

    # op_bs = (0**0.5 * (op_a @ op_b.A + op_a.A @ op_b)).expm()
    # dprint('op_bs', op_bs.space_map)

    dbs = [
        -3,
    ]
    axBall = mplfigB(Nrows=len(dbs), Ncols=2)
    for idx, db in enumerate(dbs):
        z = np.log(10 ** (db / 10.0)) / 2
        dprint(db)

        sqz_gen = z * op_a @ op_a
        sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()
        psi_sqz = sqz @ psi
        dprint("TESTING------------------")
        # dprint('op_bs', op_bs.space_map)
        dprint("psi_sqz", psi_sqz.space_map)
        # psi_sqz = op_bs @ psi_sqz
        dprint("TESTING------------------")

        psi_sqz2 = psi_sqz @ qop_fock.psi_vacuum("b", bN=bFx)
        dprint("MAP1", psi_sqz2.space_map)
        rho_sqz2 = (psi_sqz2 @ psi_sqz2.A).trace_except("a")

        psi_sqz3 = qop_fock.psi_vacuum("b", bN=bFx) @ psi_sqz
        dprint("MAP2", psi_sqz3.space_map)
        rho_sqz3 = (psi_sqz3 @ psi_sqz3.A).trace_except("a")
        # psi_sqz2.space_order(['a', 'b'])
        # psi_sqz3.space_order(['a', 'b'])
        # dprint(psi_sqz2.mat.shape, psi_sqz3.mat.shape)
        # dprint('order_check', psi_sqz2.mat - psi_sqz3.mat)

        rho_sqz = (psi_sqz @ psi_sqz.A).trace_except("a")
        dprint("SQZ_diff", rho_sqz.mat.real)
        dprint("SQZ_diff2", rho_sqz2.mat.real)
        dprint("SQZ_diff3", rho_sqz2.mat.real)

        dprint("rho_sqz", rho_sqz.space_map)
        # dprint(len(rho_sqz.space_basis), rho_sqz.space_basis[0].bC.basis, rho_sqz.space_basis[0].bR.basis)
        rho_sqz = F2Q @ rho_sqz @ F2Q.A
        # dprint(len(rho_sqz.space_basis), rho_sqz.space_basis[0].bC.basis, rho_sqz.space_basis[0].bR.basis)

        axB = mplfigB(Ncols=2)
        plot_wigner(axB.ax0_0, "a", rho=rho_sqz, bQ=bQ, lims=10)
        # axB.ax0_1.stem(abs(psi_sqz.mat[:, 0])**2, markerfmt='C0,')
        axB.ax0_1.set_xscale("log")
        axB.save(tjoin("wigner_sqz_{:.1f}db".format(-db).replace(".", "p")))
        plot_wigner(axBall["ax{}_0".format(idx)], "a", rho=rho_sqz, bQ=bQ, lims=10)
        axall = axBall["ax{}_1".format(idx)]
        # axall.stem(abs(psi_sqz.mat[:, 0])**2, markerfmt='C0,')
        axall.set_xscale("log")
    axBall.save(tjoin("wigner_sqz".format(-db)))
    return


def T_gkp_wigner_loss(plot):
    bQ = qop_fock.basis_Q(N=2 * 1024)
    bF = qop_fock.basis_Fock(N=300)
    F2Q = qop_fock.basis_change("field", bC=bF, bR=bQ)

    iDs = [2, 4, 5, 8]
    axBall = mplfigB(Nrows=len(iDs), Ncols=2)
    for idx, iD in enumerate(iDs):
        psi_gkp = F2Q.A @ qop_fock.psi_gkp("field", bN=bQ, D=1 / iD, mu=0)

        dprint("iD", iD)
        qop_fock.op_qp("field", bN=bF)

        axB = mplfigB(Ncols=2)
        plot_wigner(axB.ax0_0, "field", psi=F2Q @ psi_gkp, bQ=bQ, lims=10)
        axB.ax0_1.stem(abs(psi_gkp.mat[:, 0]) ** 2, markerfmt="C0,")
        axB.ax0_1.set_xscale("log")
        axB.save(tjoin("wigner_gkp_{:.0f}".format(iD).replace(".", "p")))
        plot_wigner(
            axBall["ax{}_0".format(idx)], "field", psi=F2Q @ psi_gkp, bQ=bQ, lims=10
        )
        axall = axBall["ax{}_1".format(idx)]
        axall.stem(abs(psi_gkp.mat[:, 0]) ** 2, markerfmt="C0,")
        axall.set_xscale("log")
    axBall.save(tjoin("wigner_gkp"))
    return


def plot_wigner(ax, *args, **kwargs):
    lims = kwargs.pop("lims", None)
    w, q, p = qop_fock.qop2wigner(*args, **kwargs)
    ax.set_aspect(1)
    minmax = np.max(abs(w))
    ax.imshow(
        w,
        extent=(q[0], q[-1], p[0], p[-1]),
        cmap="PiYG",
        vmin=-minmax,
        vmax=minmax,
        interpolation="nearest",
    )
    ax.grid(visible=False)
    if lims is not None:
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
