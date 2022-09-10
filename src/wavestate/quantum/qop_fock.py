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
from wavestate.bunch import FrozenBunch

from . import qop
from . import fock


# the trivial space
bI = qop.bI


def basis_Q(N, half_width=None):
    if half_width is None:
        half_width = (N * np.pi / 4) ** 0.5
    q = fock.linspace_clopen(half_width, N)
    dq = q[1] - q[0]
    p = fock.linspace_clopen(np.pi / dq, N)
    dp = p[1] - p[0]

    bQ = FrozenBunch(
        type="continuum",
        basis="Q",
        N=N,
        q=q,
        dq=float(dq),
        q0=float(q[0]),
        hash_ignore=("bP", "q"),
    )

    bP = FrozenBunch(
        type="continuum",
        basis="P",
        N=N,
        p=p,
        dp=float(dp),
        p0=float(p[0]),
        bQ=bQ,
        hash_ignore=("bQ", "p"),
    )
    # needed to create the cross link
    # OK as long as this happens before hash is called
    bQ._insertion_hack("bP", bP)
    return bQ


def basis_Fock(N):
    basis = FrozenBunch(
        type="continuum",
        basis="Fock",
        N=N,
    )
    return basis


def psi_vacuum(space, bN):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    psi = np.zeros((bN.N, 1), dtype=complex)
    psi[0, 0] = 1
    return qop.Operator(space, bN, bI, psi)


def rho_vacuum(space, bN):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    rho = np.zeros((bN.N, bN.N), dtype=complex)
    rho[0, 0] = 1
    return qop.Operator(space, bN, bN, rho)


def psi_gkp(space, bN, D, mu=0, s=None):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Q"
    dq = bN.dq
    if mu in [0, 1]:
        return qop.Operator(space, bN, bI, dq ** 0.5 * fock.gkp(bN.q, D=D, mu=mu, s=s))
    elif mu == "+":
        return qop.Operator(
            space,
            bN,
            bI,
            dq ** 0.5
            * (fock.gkp(bN.q, D=D, mu=0, s=s) + fock.gkp(bN.q, D=D, mu=1, s=s))
            / 2 ** 0.5,
        )
    elif mu == "-":
        return qop.Operator(
            space,
            bN,
            bI,
            dq ** 0.5
            * (fock.gkp(bN.q, D=D, mu=0, s=s) - fock.gkp(bN.q, D=D, mu=1, s=s))
            / 2 ** 0.5,
        )
    elif mu == "r":
        return qop.Operator(
            space,
            bN,
            bI,
            dq ** 0.5
            * (fock.gkp(bN.q, D=D, mu=0, s=s) + 1j * fock.gkp(bN.q, D=D, mu=1, s=s))
            / 2 ** 0.5,
        )
    elif mu == "l":
        return qop.Operator(
            space,
            bN,
            bI,
            dq ** 0.5
            * (fock.gkp(bN.q, D=D, mu=0, s=s) - 1j * fock.gkp(bN.q, D=D, mu=1, s=s))
            / 2 ** 0.5,
        )
    else:
        raise RuntimeError("Unrecognized mu")


def rho_gkp(space, bC, D, mu=0, s=None, bR=None):
    if bR is None:
        bR = bC
    # column vector, data over rows
    psiC = psi_gkp(space, bC, D=D, mu=mu, s=s)
    if hash(bR) != hash(bC):
        psiR = psi_gkp(space, bR, D=D, mu=mu, s=s).A
        return psiC @ psiR
    else:
        return psiC @ psiC.A


def psi_fock(space, bN, n_fock):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    psi = np.zeros((bN.N, 1), dtype=complex)
    psi[n_fock, 0] = 1
    return qop.Operator(space, bN, bI, psi)


def rho_fock(space, bN, n_fock):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    rho = np.zeros((bN.N, bN.N), dtype=complex)
    rho[n_fock, n_fock] = 1
    return qop.Operator(space, bN, bN, rho)


def op_ladder_up(space, bN):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    M = fock.raise_fock(bN.N)
    return qop.Operator(space, bN, bN, M)


def op_ladder_dn(space, bN):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    M = fock.lower_fock(bN.N)
    return qop.Operator(space, bN, bN, M)


def op_qp(space, bN):
    op_dn = op_ladder_dn(space, bN)
    op_up = op_dn.A
    q = (op_dn + op_up) / 2 ** 0.5
    p = 1j * (op_dn - op_up) / 2 ** 0.5
    return q, p


def op_number(space, bN):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    M = op_ladder_dn(space, bN=bN)
    return M.A @ M


def op_squeeze(space, bN, alpha):
    """
    Todo:
        Make this basis-independent
    """
    assert bN.basis == "Fock"
    M = fock.lower_fock(bN.N)
    return qop.Operator(space, bN, bN, M)


def _Fock2Q(space, bF, bQ):
    M = fock.basis_fock2q(bF.N, bQ.q) * bQ.dq ** 0.5
    return qop.Operator(space, bQ, bF, M)


def basis_change(space, bR, bC):
    if bC.basis == "Fock":
        if bR.basis == "Fock":
            return qop.Operator.bare()
        elif bR.basis == "Q":
            return _Fock2Q(space, bC, bR)
    elif bC.basis == "Q":
        if bR.basis == "Q":
            return qop.Operator.bare()
        elif bR.basis == "Fock":
            return _Fock2Q(space, bR, bC).A

    raise NotImplementedError(
        "basis change between {} and {} is not supported".format(bC, bR)
    )


def qop2wigner(space, rho=None, psi=None, bQ=None, method="hermitian"):
    if rho is None and psi is None:
        raise RuntimeError("Must specify either rho or psi")
    if rho is not None and psi is not None:
        raise RuntimeError("Must specify only one of rho or psi")
    if psi is not None:
        op = psi @ psi.A
    else:
        op = rho

    bR, bC = op.basis(space)
    if bQ is not None:
        if hash(bR) == hash(bQ):
            if bR.basis == "Fock":
                F2Q = basis_change(space, bC=bR, bR=bQ)
                op = F2Q @ op @ F2Q.A
            elif hash(bR) == hash(bQ):
                pass
            else:
                raise RuntimeError("Unknown basis")
        else:
            if bR.basis == "Fock":
                F2Q = basis_change(space, bC=bR, bR=bQ)
                op = F2Q @ op
            else:
                raise RuntimeError("Unknown basis")

            if bC.basis == "Fock":
                Q2F = basis_change(space, bR=bC, bC=bQ)
                op = op @ Q2F
            else:
                raise RuntimeError("Unknown basis")
    else:
        assert bR.basis == "q"
        assert bC.basis == "q"
        assert bC.N == bQ.N

    bR, bC = op.basis(space)
    op2 = op.trace_except(space)
    w, p = fock.rhoq2wigner_fft(op2.mat, bC.q, method=method)
    return w, bC.q, p
