# -*- coding: utf-8 -*-
"""
"""
from __future__ import division, print_function, unicode_literals
import numpy as np


def matrix_stack(arr, dtype = None, **kwargs):
    """
    This routing allows one to construct 2D matrices out of heterogeneously
    shaped inputs. it should be called with a list, of list of np.array objects
    The outer two lists will form the 2D matrix in the last two axis, and the
    internal arrays will be broadcasted to allow the array construction to
    succeed

    example

    matrix_stack([
        [np.linspace(1, 10, 10), 0],
        [2, np.linspace(1, 10, 10)]
    ])

    will create an array with shape (10, 2, 2), even though the 0, and 2
    elements usually must be the same shape as the inputs to an array.

    This allows using the matrix-multiply "@" operator for many more
    constructions, as it multiplies only in the last-two-axis. Similarly,
    np.linalg.inv() also inverts only in the last two axis.
    """
    Nrows = len(arr)
    Ncols = len(arr[0])
    vals = []
    dtypes = []
    for r_idx, row in enumerate(arr):
        assert(len(row) == Ncols)
        for c_idx, kdm in enumerate(row):
            kdm = np.asarray(kdm)
            vals.append(kdm)
            dtypes.append(kdm.dtype)

    #dt = np.find_common_type(dtypes, ())
    if dtype is None:
        dtype = np.result_type(*vals)
    bc = broadcast_deep(vals)

    if len(bc.shape) == 0:
        return np.array(arr)

    Marr = np.empty(bc.shape + (Nrows, Ncols), dtype = dtype, **kwargs)
    #print(Marr.shape)

    for r_idx, row in enumerate(arr):
        for c_idx, kdm in enumerate(row):
            Marr[..., r_idx, c_idx] = kdm
    return Marr


def broadcast_deep(mlist):
    """
    Performs the same operation as np.broadcast, but does not use *args
    (takes a list of numpy arrays instead) it also can operate on arbitrarily
    long lists (rather than be limited by 32). The partial ordering on
    dtype broadcasting allows this algorithm to be recursive.
    """
    nlist = []
    for idx in range((len(mlist) + 31) // 32):
        bc = np.broadcast(*mlist[idx * 32 : (idx+1)*32])
        nlist.append(bc)

    if len(nlist) == 1:
        return nlist[0]

    nd = np.array([1])
    while len(nlist) > 1:
        blist = [np.broadcast_to(nd, bc.shape) for bc in nlist if bc.shape != ()]
        if not blist:
            #then they must all be null shapes
            return nlist[0]
        nlist = []
        for idx in range((len(blist) + 31) // 32):
            bc = np.broadcast(*blist[idx * 32 : (idx+1)*32])
            nlist.append(bc)
    return nlist[0]


def broadcast_shapes(shapes):
    """
    Finds the common shape of a list of arrays, such that broadcasting into
    that shape will succeed.
    """
    nd = np.array([1])

    nlist = [np.broadcast_to(nd, shape) for shape in shapes if shape != ()]
    if not nlist:
        return ()

    while len(nlist) > 1:
        blist = [np.broadcast_to(nd, bc.shape) for bc in nlist if bc.shape != ()]
        if not blist:
            #then they must all be null shapes
            return nlist[0]
        nlist = []
        for idx in range((len(blist) + 31) // 32):
            bc = np.broadcast(*blist[idx * 32 : (idx+1)*32])
            nlist.append(bc)
    return nlist[0].shape


def matrix_stack_id(arr, **kwargs):
    arrs = []
    for idx, a in enumerate(arr):
        lst = [0] * len(arr)
        lst[idx] = a
        arrs.append(lst)
    return matrix_stack(arrs, **kwargs)
