# coding: utf-8
"""
"""
import numpy as np
from wavestate.bunch import FrozenBunch

from . import qop

Mpauli_x = np.array(
    [[0, 1],
     [1, 0]]
)
Mpauli_y = np.array(
    [[0, -1j],
     [1j, 0]]
)
Mpauli_z = np.array(
    [[1, 0],
     [0, -1]]
)
Meye2 = np.array(
    [[1, 0],
     [0, 1]]
)

bQz = FrozenBunch(
    type = 'qubit',
    basis = 'z',
    N = 2,
)

bQx = FrozenBunch(
    type = 'qubit',
    basis = 'y',
    N = 2,
)

bQy = FrozenBunch(
    type = 'qubit',
    basis = 'x',
    N = 2,
)


def pauli_x(space, bQ = bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_x)

def pauli_y(space, bQ = bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_y)

def pauli_z(space, bQ = bQz):
    return qop.Operator(space, bQ, bQ, Mpauli_z)

def id(space, bQ = bQz):
    return qop.Operator(space, bQ, bQ, Meye2)
