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

Mpauli_x = np.array([[0, 1], [1, 0]])
Mpauli_y = np.array([[0, -1j], [1j, 0]])
Mpauli_z = np.array([[1, 0], [0, -1]])
Meye2 = np.array([[1, 0], [0, 1]])

bQz = FrozenBunch(
    type="qubit",
    basis="z",
    N=2,
)

bQx = FrozenBunch(
    type="qubit",
    basis="y",
    N=2,
)

bQy = FrozenBunch(
    type="qubit",
    basis="x",
    N=2,
)


def pauli_x(space, bQ=bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_x)


def pauli_y(space, bQ=bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_y)


def pauli_z(space, bQ=bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_z)


def id(space, bQ=bQz):
    return qop.Operator(space, bQ, bQ, Meye2)
