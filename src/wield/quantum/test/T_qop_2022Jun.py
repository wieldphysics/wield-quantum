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

from wield.quantum import ( # noqa
    qop,
    qop_fock,
    qop_qubit,
    fock
)

from wield.utilities.mpl import (  # noqa
    # generate_stacked_plot_ax,
    mplfigB,
    asavefig,
)

from wield.pytest.fixtures import (  # noqa: F401
    tpath_join,
    dprint,
    plot
)


def matmul_many(mats):
    m = mats[0]
    for other in mats[1:]:
        m = m @ other
    return m

def sum_many(mats, eye):
    m = mats[0] @ eye
    for other in mats[1:]:
        m = m + (other @ eye)
    return m


def T_atom_optics_bits(tpath_join, dprint, plot):
    L = 0.5

    bF = qop_fock.basis_Fock(N=10)
    op_a = qop_fock.op_ladder_dn("a", bN=bF)

    eyes = []
    bits = []
    rho_ns = []
    rho_ps = []
    for N in range(3):
        bit = qop_qubit.pauli_z('atom{}'.format(N))
        bits.append(bit)
        eyes.append(qop_qubit.id('atom{}'.format(N)))
        rho_ns.append(qop_qubit.rho_n('atom{}'.format(N)))
        rho_ps.append(qop_qubit.rho_p('atom{}'.format(N)))
    eye = matmul_many(eyes)
    #eye.segments_condense()
    print("EYE: ", eye.mat)

    b = sum_many(bits, eye)
    #b.segments_condense()
    print(b.mat)
    #print(b.space_map)
    #print(b.index_order)

    rho_n = matmul_many(rho_ns)
    rho_p = matmul_many(rho_ps)
    rho_n.segments_condense()
    print((rho_p).mat)

    print((rho_p @ b).trace_except().mat)

    if False:
        sqz_gen = z * op_a @ op_a
        sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()

    return


def T_atom_optics_bits2(tpath_join, dprint, plot):
    L = 0.5

    bF = qop_fock.basis_Fock(N=10)
    op_a = qop_fock.op_ladder_dn("a", bN=bF)

    bits = qop.empty0
    rho_ns = qop.empty1
    rho_ps = qop.empty1
    psi_ns = qop.empty1
    psi_ps = qop.empty1
    for N in range(10):
        rho_ns = rho_ns @ qop_qubit.rho_n('atom{}'.format(N))
        rho_ps = rho_ps @ qop_qubit.rho_p('atom{}'.format(N))
        psi_ns = psi_ns @ qop_qubit.psi_n('atom{}'.format(N))
        psi_ps = psi_ps @ qop_qubit.psi_p('atom{}'.format(N))
        bits = bits + qop_qubit.pauli_z('atom{}'.format(N)) * 0.5

    b = bits
    b.segments_condense()
    print(b.mat)

    rho_n = rho_ns
    rho_p = rho_ps
    rho_n = psi_ns @ psi_ns.A
    rho_p = psi_ps @ psi_ps.A
    rho_n.segments_condense()
    print((rho_p).mat)

    print((rho_p @ b).trace())

    if False:
        sqz_gen = z * op_a @ op_a
        sqz_gen = sqz_gen - sqz_gen.A
        sqz = sqz_gen.expm()

    return
