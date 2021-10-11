# coding: utf-8
"""
"""
import numpy as np
from declarative.bunch import FrozenBunch

#asavefig.formats.png.use = True
import itertools

from transient.pytest import (  # noqa: F401
    ic, tpath_join, pprint, plot
)

from transient.quantum import qop

pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
eye2 = np.array([[1, 0], [0, 1]])

def T_qop1(tpath_join, pprint, plot):

    basis = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 2,
    )

    op = qop.Operator.bare()
    pprint(op.mat)
    op.join('bit1', basis, basis, pauli_z)
    pprint(op.mat)
    op.join('bit2', basis, basis, pauli_x)
    pprint(op.mat)
    op.join('bit3', basis, basis, eye2)
    pprint(op.mat)

    pprint(op.space_map)
    pprint(op.space_segments)
    pprint(op.index_order)
    op.last_segment(['bit1'])
    op.last_segment(['bit2', 'bit3', 'bit1'])
    #pprint(op.space_basis)

    #now test that the reordered matrix is equal to one built natively in the
    #requested order 
    op2 = qop.Operator.bare()
    op2.join('bit2', basis, basis, pauli_x)
    op2.join('bit3', basis, basis, eye2)
    op2.join('bit1', basis, basis, pauli_z)
    #pprint(op2.space_basis)
    pprint(op.mat == op2.mat)
    assert(op == op2)
    return


def T_qop2(tpath_join, pprint, plot):
    op = qop.Operator.bare()
    pprint(op.mat)

    basis = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 2,
    )
    op.join('bit1', basis, basis, pauli_z)
    op.join('bit2', basis, basis, pauli_x)

    A = op.mat.reshape(2,2,1,1,2,2)
    B = op.mat.reshape(1,1,2,2,2,2)
    C = A @ B
    pprint(C.shape)
    return


def T_qop3A(tpath_join, pprint, plot):
    """
    test multiplication of qubit spaces. This test does not incorporate a common space
    """

    basis = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 2,
    )

    op1 = qop.Operator('bit1', basis, basis, pauli_z)
    op2 = qop.Operator('bit2', basis, basis, pauli_x)

    prod12 = op1 @ op2
    op12 = qop.Operator.bare()
    op12.join('bit1', basis, basis, pauli_z)
    op12.join('bit2', basis, basis, pauli_x)
    #op12.join('bit3', basis, basis, eye2)
    pprint(prod12.mat == op12.mat)
    assert(np.all(prod12.mat == op12.mat))

    op3 = qop.Operator('bit3', basis, basis, eye2)
    return

def T_qop3B(tpath_join, pprint, plot):
    """
    test multiplication of qubit spaces. This test does incorporates a common space
    """

    basis = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 2,
    )

    op1 = qop.Operator('bit1', basis, basis, pauli_z)
    op1.join('bit3', basis, basis, pauli_y)
    op2 = qop.Operator('bit2', basis, basis, pauli_x)
    op2.join('bit3', basis, basis, pauli_y)

    prod12 = op1 @ op2
    op12 = qop.Operator.bare()
    op12.join('bit1', basis, basis, pauli_z)
    op12.join('bit2', basis, basis, pauli_x)
    op12.join('bit3', basis, basis, pauli_y @ pauli_y)
    #op12.join('bit3', basis, basis, eye2)
    pprint(prod12.mat == op12.mat)
    assert(np.all(prod12.mat == op12.mat))
    return


def T_qop_ordering(tpath_join, pprint, plot):
    """
    test multiplication of qubit spaces. This test does incorporates a common space
    """

    basis1 = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 1,
    )
    basis2 = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 2,
    )
    basis3 = FrozenBunch(
        type = 'qubit',
        basis = 'z',
        N = 3,
    )

    op1 = qop.Operator('bit1', basis2, basis1, np.array([[2,3]]).T)
    op2 = qop.Operator('bit2', basis3, basis1, np.array([[5, 7,11]]).T)

    prod1 = (op1 @ op2)
    pprint(prod1.mat)

    prod2 = (op2 @ op1)
    pprint(prod2.mat)
    pprint('-------order------')
    prod1.last_segment(['bit1', 'bit2'])
    prod2.last_segment(['bit1', 'bit2'])
    pprint(prod1.mat)
    pprint(prod2.mat)

    pprint('-----again-----------')
    prod1 = (op1 @ op2)
    prod1 = prod1 @ prod1.A
    pprint(prod1.mat)

    prod2 = (op2 @ op1)
    prod2 = prod2 @ prod2.A
    pprint(prod2.mat)
    pprint('-------order------')
    prod1.last_segment(['bit1', 'bit2'])
    prod2.last_segment(['bit1', 'bit2'])
    pprint(prod1.mat)
    pprint(prod2.mat)
    return
