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

from wield.utilities.mpl import (  # noqa
    # generate_stacked_plot_ax,
    mplfigB,
    asavefig,
)

from wield.pytest import tpath_join, dprint, plot  # noqa: F401

from wield.quantum import qop, qop_fock, qop_qubit, fock  # noqa


def T_qop_fockdisp(tpath_join, dprint, plot):
    bF = qop_fock.basis_Fock(N=30)
    bQ = qop_fock.basis_Q(half_width=30, N=2048)
    psi = qop_fock.psi_vacuum("field", bN=bF)
    dprint(psi.nR, psi.nC, psi.mat.shape)

    toQ = qop_fock.basis_change("field", bQ, bF)
    dprint(toQ.nR, toQ.nC, toQ.mat.shape)

    def plot_psi(psi, name=None, line=None):
        psi_q = toQ @ psi
        psi_q = psi_q.mat[:, 0]
        axB = mplfigB(Nrows=3)
        axB.ax0.plot(bQ.q, bQ.dq * np.cumsum(abs(psi_q) ** 2))
        axB.ax1.plot(bQ.q, abs(psi_q))
        axB.ax2.plot(bQ.q, fock.angle(psi_q))
        if line is not None:
            axB.ax0.axvline(line)
        if name is not None:
            axB.save(tpath_join(name))

    plot_psi(psi, "vac")

    opA = qop_fock.op_ladder_dn("field", bN=bF)

    alpha = np.array(2)
    disp = (alpha.conj() * opA.A - alpha * opA).expm()
    factor = 2 ** 0.5

    psi_disp = disp @ psi
    plot_psi(psi_disp, "disp", line=abs(alpha) * factor)

    opN = qop_fock.op_number("field", bN=bF)
    dprint((psi_disp.A @ opN @ psi_disp).mat)

    return
    # this tests SQZ
    z = np.array(0.5)
    aM2 = opA @ opA
    dprint(aM2)
    sqz = expm(z.conj() * aM2 - z * adj(aM2))
    dprint(sqz @ psi)

    plot_psi(sqz @ psi, "sqz")
    return


def T_qop_fockdisp2(tpath_join, dprint, plot):
    bF = qop_fock.basis_Fock(N=50)
    opA = qop_fock.op_ladder_dn("field", bN=bF)

    alpha = np.array(2)
    disp = (alpha.conj() * opA.A - alpha * opA).expm()

    aM2 = disp.A @ opA @ disp - opA
    dprint(np.diag(aM2.mat))

    aMA2 = disp.A @ opA.A @ disp - opA.A
    dprint(np.diag(aMA2.mat))
    return


def T_qop_gkp_norm(tpath_join, dprint, plot):
    """
    This test makes sure normalizations are right and that the Fourier transform
    of a GKP state is correct
    """
    D = 1 / 5
    bQ = qop_fock.basis_Q(half_width=32, N=32 ** 2)
    psi = qop_fock.psi_gkp("field", bN=bQ, D=D, mu=0)
    dprint(psi.nR, psi.nC, psi.mat.shape)
    # the Fourier transform of a GKP is in the superposition of |0> and |1>
    psi_p2 = (
        fock.gkp(q=bQ.bP.p, D=D, mu=0) + fock.gkp(q=bQ.bP.p, D=D, mu=1)
    ) / 2 ** 0.5

    psi_q = psi.mat[:, 0]
    axB = mplfigB(Nrows=3)
    axB.ax0.plot(bQ.q, bQ.dq * np.cumsum(abs(psi_q) ** 2))
    axB.ax1.semilogy(bQ.q, abs(psi_q))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_q)))
    axB.ax2.plot(bQ.q, fock.angle(psi_q))

    axB.save(tpath_join("gkp_q"))

    axB = mplfigB(Nrows=3)
    psi_p, p = fock.q2p(psi_q, bQ.q, extend=True)
    dprint(bQ.bP.p, p)
    dp = p[1] - p[0]
    axB.ax0.plot(p, dp * np.cumsum(abs(psi_p) ** 2))
    axB.ax1.semilogy(p, abs(psi_p))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_p)))
    axB.ax2.plot(p, fock.angle(psi_p))

    axB.ax0.plot(bQ.bP.p, bQ.bP.dp * np.cumsum(abs(psi_p2) ** 2))
    axB.ax1.semilogy(bQ.bP.p, abs(psi_p2))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_p2)))
    axB.ax2.plot(bQ.bP.p, fock.angle(psi_p2))

    # axB.ax0.plot(bQ.q, bQ.dq*np.cumsum(abs(psi_q)**2))
    # axB.ax1.semilogy(bQ.q, abs(psi_q))
    # axB.ax1.set_ylim(1e-10, np.max(abs(psi_q)))
    # axB.ax2.plot(bQ.q, fock.angle(psi_q))

    axB.save(tpath_join("gkp_p"))


def T_qop_gkp_norm2(tpath_join, dprint, plot):
    """
    This test makes sure normalizations are right and that the Fourier transform
    of a GKP state is correct
    """
    D = 1 / 5
    bQ = qop_fock.basis_Q(half_width=32, N=32 ** 2)
    psi = qop_fock.psi_gkp("field", bN=bQ, D=D, mu="r")
    dprint(psi.nR, psi.nC, psi.mat.shape)
    # the Fourier transform of a GKP is in the superposition of |0> and |1>
    psi_p2 = (
        np.exp(np.pi / 4 * 1j)
        * (fock.gkp(q=bQ.bP.p, D=D, mu=0) - 1j * fock.gkp(q=bQ.bP.p, D=D, mu=1))
        / 2 ** 0.5
    )

    psi_q = psi.mat[:, 0]
    axB = mplfigB(Nrows=3)
    axB.ax0.plot(bQ.q, bQ.dq * np.cumsum(abs(psi_q) ** 2))
    axB.ax1.semilogy(bQ.q, abs(psi_q))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_q)))
    axB.ax2.plot(bQ.q, fock.angle(psi_q))

    axB.save(tpath_join("gkp_q"))

    axB = mplfigB(Nrows=3)
    psi_p, p = fock.q2p(psi_q, bQ.q, extend=True)
    dprint(bQ.bP.p, p)
    dp = p[1] - p[0]
    axB.ax0.plot(p, dp * np.cumsum(abs(psi_p) ** 2))
    axB.ax1.semilogy(p, abs(psi_p))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_p)))
    axB.ax2.plot(p, fock.angle(psi_p))

    axB.ax0.plot(bQ.bP.p, bQ.bP.dp * np.cumsum(abs(psi_p2) ** 2))
    axB.ax1.semilogy(bQ.bP.p, abs(psi_p2))
    axB.ax1.set_ylim(1e-10, np.max(abs(psi_p2)))
    axB.ax2.plot(bQ.bP.p, fock.angle(psi_p2))

    # axB.ax0.plot(bQ.q, bQ.dq*np.cumsum(abs(psi_q)**2))
    # axB.ax1.semilogy(bQ.q, abs(psi_q))
    # axB.ax1.set_ylim(1e-10, np.max(abs(psi_q)))
    # axB.ax2.plot(bQ.q, fock.angle(psi_q))

    axB.save(tpath_join("gkp_p"))


def T_gkp_wigner(tpath_join, dprint, plot):
    D = 1 / 4
    for mu in [0, 1, "+", "-", "l", "r"]:
        bQ = qop_fock.basis_Q(N=4 * 1024)
        rho = qop_fock.rho_gkp("field", bC=bQ, D=D, mu=mu)
        # the Fourier transform of a GKP is in the superposition of |0> and |1>
        axB = mplfigB()
        plot_wigner(axB.ax0, "field", rho=rho, bQ=bQ, lims=10)
        axB.save(tpath_join("wigner_{}".format(mu)))
    return


def T_gkp_wigner_span(tpath_join, dprint, plot):
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
        axB.ax0_1.stem(
            abs(psi_gkp.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axB.ax0_1.set_xscale("log")
        axB.save(tpath_join("wigner_gkp_{:.0f}".format(iD).replace(".", "p")))
        plot_wigner(
            axBall["ax{}_0".format(idx)], "field", psi=F2Q @ psi_gkp, bQ=bQ, lims=10
        )
        axall = axBall["ax{}_1".format(idx)]
        axall.stem(
            abs(psi_gkp.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axall.set_xscale("log")
    axBall.save(tpath_join("wigner_gkp"))
    return


def T_sqz_wigner(tpath_join, dprint, plot):
    D = 1 / 4
    bQ = qop_fock.basis_Q(N=2 * 1024)
    bF = qop_fock.basis_Fock(N=300)
    F2Q = qop_fock.basis_change("field", bC=bF, bR=bQ)

    psi = F2Q.A @ qop_fock.psi_gkp("field", bN=bQ, D=D, mu=0)
    psi = qop_fock.psi_vacuum("field", bN=bF)

    op_a = qop_fock.op_ladder_dn("field", bN=bF)

    dbs = [-3]  # , -6, -9, -10]
    axBall = mplfigB(Nrows=len(dbs), Ncols=2)
    for idx, db in enumerate(dbs):
        z = np.log(10 ** (db / 10.0)) / 2
        dprint(db)

        sqz_gen = z * op_a @ op_a
        dprint("sqz:-----", sqz_gen.mat.real)
        dprint("sqz:-----", (sqz_gen.A).mat.real)
        dprint("sqz:-----", (sqz_gen.mat - (sqz_gen.A).mat).real)
        sqz_gen = sqz_gen - (sqz_gen.A)
        dprint("sqz:-----", sqz_gen.mat)
        sqz = sqz_gen.expm()
        psi_sqz = sqz @ psi

        qop_fock.op_qp("field", bN=bF)

        axB = mplfigB(Ncols=2)
        plot_wigner(axB.ax0_0, "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=10)
        axB.ax0_1.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axB.ax0_1.set_xscale("log")
        axB.save(tpath_join("wigner_sqz_{:.1f}db".format(-db).replace(".", "p")))
        plot_wigner(
            axBall["ax{}_0".format(idx)], "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=10
        )
        axall = axBall["ax{}_1".format(idx)]
        axall.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axall.set_xscale("log")
    axBall.save(tpath_join("wigner_sqz".format(-db)))
    return


def T_shear_wigner(tpath_join, dprint, plot):
    D = 1 / 2
    bQ = qop_fock.basis_Q(N=1 * 1024)
    bF = qop_fock.basis_Fock(N=100)
    F2Q = qop_fock.basis_change("field", bC=bF, bR=bQ)

    psi = qop_fock.psi_vacuum("field", bN=bF)
    # psi = F2Q.A @ qop_fock.psi_gkp('field', bN=bQ, D=D, mu=0)

    op_a = qop_fock.op_ladder_dn("field", bN=bF)
    op_q, op_p = qop_fock.op_qp("field", bN=bF)
    op_n = qop_fock.op_number("field", bN=bF)

    dbs = [-3, -6, -9, -10]
    axBall = mplfigB(Nrows=len(dbs), Ncols=2)
    for idx, db in enumerate(dbs):
        z = 1j * np.log(10 ** (db / 10.0)) / 2
        dprint(db, z)

        sqz_gen = z * op_q @ op_q
        # sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()
        psi_sqz = sqz @ psi

        qop_fock.op_qp("field", bN=bF)

        axB = mplfigB(Ncols=2)
        plot_wigner(axB.ax0_0, "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=5)
        axB.ax0_1.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axB.ax0_1.set_xscale("log")
        axB.save(tpath_join("wigner_sqz_{:.1f}db".format(-db).replace(".", "p")))
        plot_wigner(
            axBall["ax{}_0".format(idx)], "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=5
        )
        axall = axBall["ax{}_1".format(idx)]
        axall.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axall.set_xscale("log")
    axBall.save(tpath_join("wigner_sqz".format(-db)))
    return


def T_shear_gkp_wigner(tpath_join, dprint, plot):
    D = 1 / 4
    bQ = qop_fock.basis_Q(N=2 * 1024)
    bF = qop_fock.basis_Fock(N=300)
    F2Q = qop_fock.basis_change("field", bC=bF, bR=bQ)

    psi = qop_fock.psi_gkp("field", bN=bQ, D=D, mu=0)
    dprint("SUM", (psi @ psi.A).trace_except().mat)
    psi = F2Q.A @ psi

    op_a = qop_fock.op_ladder_dn("field", bN=bF)
    op_q, op_p = qop_fock.op_qp("field", bN=bF)

    dbs = [-3, -6, -9]
    axBall = mplfigB(Nrows=len(dbs), Ncols=2)
    for idx, db in enumerate(dbs):
        z = 1j * np.log(10 ** (db / 10.0)) / 2
        dprint(db)

        sqz_gen = z * op_q @ op_q
        sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()
        psi_sqz = sqz @ psi

        qop_fock.op_qp("field", bN=bF)
        dprint("SUM", (psi_sqz @ psi_sqz.A).trace_except().mat)

        axB = mplfigB(Ncols=2)
        plot_wigner(axB.ax0_0, "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=15)
        axB.ax0_1.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axB.ax0_1.set_xscale("log")
        axB.save(tpath_join("wigner_sqz_{:.1f}db".format(-db).replace(".", "p")))
        plot_wigner(
            axBall["ax{}_0".format(idx)], "field", psi=F2Q @ psi_sqz, bQ=bQ, lims=15
        )
        axall = axBall["ax{}_1".format(idx)]
        axall.stem(
            abs(psi_sqz.mat[:, 0]) ** 2, markerfmt="C0,", use_line_collection=True
        )
        axall.set_xscale("log")
    axBall.save(tpath_join("wigner_sqz".format(-db)))
    return


def T_basis_change(tpath_join, dprint, plot):
    bQ = qop_fock.basis_Q(N=2 * 1024)
    bF = qop_fock.basis_Fock(N=300)
    F2Q = qop_fock.basis_change("field", bC=bF, bR=bQ)
    I = F2Q.A @ F2Q
    dprint(I.mat)


def T_qop_gkp(tpath_join, dprint, plot):
    bF = qop_fock.basis_Fock(N=30)
    bQ = qop_fock.basis_Q(half_width=30, N=2048)
    psi = qop_fock.psi_gkp("field", bN=bQ, D=4, mu=0)
    dprint(psi.nR, psi.nC, psi.mat.shape)

    toQ = qop_fock.basis_change("field", bQ, bF)
    dprint(toQ.nR, toQ.nC, toQ.mat.shape)

    def plot_psi(psi, name=None, line=None):
        if psi.basis("field").bC.basis == "Fock":
            psi_q = toQ @ psi
            psi_q = psi_q.mat[:, 0]
        else:
            psi_q = psi.mat[:, 0]
        axB = mplfigB(Nrows=3)
        axB.ax0.plot(bQ.q, bQ.dq * np.cumsum(abs(psi_q) ** 2))
        axB.ax1.semilogy(bQ.q, abs(psi_q))
        axB.ax1.set_ylim(1e-10, np.max(abs(psi_q)))
        axB.ax2.plot(bQ.q, fock.angle(psi_q))
        if line is not None:
            axB.ax0.axvline(line)
        if name is not None:
            axB.save(tpath_join(name))

    plot_psi(psi, "gkp")
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
    ax.grid(b=False)
    if lims is not None:
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
