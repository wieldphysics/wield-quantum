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
import numbers
from scipy.linalg import expm

# asavefig.formats.png.use = True
import collections
from wavestate.bunch import FrozenBunch
import itertools

import copy

# the trivial space. Useful for making vectors
bI = FrozenBunch(
    type="trivial",
    basis="scalar",
    N=1,
)


# Type for internal storage of the space descriptions for a named space
SpaceBasis = collections.namedtuple("SpaceBasis", ["bR", "bC"])

# This type holds basic values needed to determine what represents an index
# of a matrix. These are stored in a list of tuples, where each tuple
# contains several of these indexes. 
# Space: the name of the space
# col: bool indicating if the index is a column (True) or a row (False)
# N: the size of the space associated with this index
SpaceIndex = collections.namedtuple("SpaceIndex", ["space", "col", "N"])


class Operator(object):
    """ """

    bI = bI

    def _init_empty(self, value=1, dtype=complex):
        """
        Special method that causes initialization. Used for "bare" construction
        as well as the standard user-available __init__
        """

        # the matrix is stored in "unraveled" form
        self._matrix = value * np.ones((), dtype=dtype)

        # contains tuples of (row, col) basis containers
        self.space_map = {}
        self.nC = 1
        self.nR = 1

        # list of tuples of SpaceIndex(space, col, N)
        self.index_order = []

        # list of tuples of segments
        self.space_segments = []
        return

    @property
    def mat(self):
        self.segments_condense()
        return self._matrix

    def __init__(self, space, row_basis, col_basis, M):
        self._init_empty()
        assert isinstance(row_basis, FrozenBunch)
        assert isinstance(col_basis, FrozenBunch)
        self.join(space, row_basis, col_basis, M)
        return

    @property
    def spaces(self):
        return self.space_map.keys()

    @classmethod
    def bare(cls, value):
        self = cls.__new__(cls)
        self._init_empty(value=value)
        return self

    # def __getitem__(self, space):
    #    return self.space_map[space]

    def basis(self, space):
        return self.space_map[space]

    def copy(self):
        n = self.__class__.__new__(self.__class__)
        n._matrix = self._matrix
        n.space_map = copy.copy(self.space_map)
        n.space_segments = copy.copy(self.space_segments)
        n.index_order = copy.copy(self.index_order)
        n.nR = self.nR
        n.nC = self.nC
        return n

    def join(self, space, row_basis, col_basis, M):
        assert space not in self.space_map
        nR = row_basis.N
        nC = col_basis.N
        self.space_map[space] = SpaceBasis(row_basis, col_basis)

        self.index_order.append(SpaceIndex(space=space, col=False, N=nR))
        self.index_order.append(SpaceIndex(space=space, col=True, N=nC))
        self.space_segments.append((space,))

        # outer product
        self._matrix = M.reshape(1, 1, nR, nC) * self._matrix.reshape(self.nR, self.nC, 1, 1)
        self.nR *= nR
        self.nC *= nC
        self._matrix = self._matrix.reshape(self.nR, self.nC)
        return

    def join_id(self, space_row_col):
        if isinstance(space_row_col, dict):
            src = []
            for name, (r, c) in space_row_col.items():
                src.append((name, r, c))
            space_row_col = src

        N = 1
        for space, row_basis, col_basis in space_row_col:
            assert(row_basis == col_basis)
            N *= row_basis.N

        return self.join_many(space_row_col, M=np.eye(N))

    def join_many(self, space_row_col, M):
        nR = 1
        nC = 1
        index_order_post = []
        spaces = []
        for space, row_basis, col_basis in space_row_col:
            spaces.append(space)
            assert space not in self.space_map
            nR *= row_basis.N
            nC *= col_basis.N
            self.space_map[space] = SpaceBasis(row_basis, col_basis)

            self.index_order.append(SpaceIndex(space=space, col=False, N=row_basis.N))
            index_order_post.append(SpaceIndex(space=space, col=True, N=col_basis.N))
        self.index_order.extend(index_order_post)
        self.space_segments.append(tuple(spaces))

        # outer product
        self._matrix = M.reshape(1, 1, nR, nC) * self._matrix.reshape(self.nR, self.nC, 1, 1)
        self.nR *= nR
        self.nC *= nC
        self._matrix = self._matrix.reshape(self.nR, self.nC)
        return

    def _join_many(self, spaces, row_bases, col_bases, M):
        return self.join_many(zip(spaces, row_bases, col_bases))

    def segments_set(self, space_segments):
        """
        Takes the list of spaces and creates a last segment from them, pulling from
        other segments
        """
        new_segments = space_segments
        # make sure all of the spaces are represented
        assert set(itertools.chain(*new_segments)) == set(
            itertools.chain(*self.space_segments)
        )

        new_index_order = []
        for segment in new_segments:
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=False, N=self.space_map[space].bR.N)
                )
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=True, N=self.space_map[space].bC.N)
                )
        # print('prev', self.index_order)
        # print('new', new_index_order)

        pmap = {v: i for i, v in enumerate(self.index_order)}
        trans_map = [pmap[v] for v in new_index_order]
        # print('trans_map', trans_map)

        if np.all(trans_map == np.arange(len(trans_map))):
            # do nothing!
            return

        shape_map = [v.N for v in self.index_order]
        self._matrix = (
            self._matrix.reshape(*shape_map).transpose(trans_map).reshape(self.nR, self.nC)
        )
        self.index_order = new_index_order
        self.space_segments = new_segments
        return

    def segments_condense(self):
        self.last_segment(tuple(sorted(itertools.chain(*self.space_segments))))

    @property
    def mat_ordered(self):
        return self._matrix.reshape(*[i.N for i in self.index_order])

    def last_segment(self, spaces, collect=True):
        """
        Takes the list of spaces and creates a last segment from them, pulling from
        other segments
        """
        # print(self.space_segments)
        new_segments = []
        for segment in self.space_segments:
            new_seg = tuple(space for space in segment if space not in spaces)
            if new_seg:
                new_segments.append(new_seg)
        if collect:
            old_segments = new_segments
            if old_segments:
                new_segments = [tuple(itertools.chain(*old_segments))]
        new_segments.append(tuple(spaces))
        # print(new_segments)

        self.segments_set(new_segments)
        return

    def trace(self):
        return self.trace_spaces(*self.space_map.keys()).mat[..., 0, 0]

    def trace_except(self, *spaces):
        return self.trace_spaces(*set(self.space_map.keys()) - set(spaces))

    def trace_spaces(self, *spaces):
        if not spaces:
            return self.copy()

        spaces = sorted(spaces, key=lambda k: self.space_map[k])
        # print('map1', self.space_map)
        # print(self._matrix)
        self.last_segment(spaces)
        # print('map2', self.space_map)
        # print(self._matrix)
        Ntrace = len(spaces)
        # Nremain = len(self.space_map) - Ntrace
        # print("trace, remain", spaces, Ntrace, Nremain, self.space_map)
        nR = 1
        nC = 1
        new_space_map = dict(self.space_map)
        for space in spaces:
            new_space_map.pop(space)
            bR, bC = self.space_map[space]
            nR *= bR.N
            nC *= bC.N
        mat = self._matrix.reshape(self.nR // nR, self.nC // nC, nR, nC)
        # print("HEYY", mat.shape)
        # if mat.shape[2:] == (2,2):
        #    print(mat[:,:,0,0])
        #    print(mat[:,:,0,1])
        #    print(mat[:,:,1,0])
        #    print(mat[:,:,1,1])
        mat = np.trace(mat, axis1=2, axis2=3)
        # print(mat)
        self.nR = self.nR // nR
        self.nC = self.nC // nC

        n = self.__class__.__new__(self.__class__)
        n._matrix = mat
        n.nR = nR
        n.nC = nC
        n.space_map = new_space_map
        n.space_segments = self.space_segments[:-1]
        n.index_order = self.index_order[: len(self.index_order) - 2 * len(spaces)]
        return n

    def __matmul__(self, other):
        ospaces = set(other.space_map)
        sspaces = set(self.space_map)
        # print('ospaces', ospaces)
        # print('sspaces', sspaces)

        # at some point the keys may not be consistently sortable
        # maybe just sort on the hash, just for the consistency to minimize reordering
        cspaces = set(sspaces.intersection(ospaces))
        # cspaces = list(reversed(sorted(sspaces.intersection(ospaces))))

        # num = len(cspaces)
        # print('cspaces', cspaces)

        # if they are identical, then save some transposes
        # TODO, could make this more efficient yet
        if cspaces:
            if (set(self.space_segments[-1]) == cspaces) and (
                self.space_segments[-1] == other.space_segments[-1]
            ):
                # don't need to do anything, as they are in the same order
                # but go ahead and use the existing ordering
                cspaces = self.space_segments[-1]
            else:
                cspaces = sorted(cspaces)
                self.last_segment(cspaces)
                other.last_segment(cspaces)

            # at this point, both objects have their last segment as the common spaces
            # so count the elements in that segment

            nR = 1
            nI = 1
            nC = 1
            # this loop is going backwards, but that is OK for this counting purpose
            cspace_map = {}
            # for rows
            cindex_order = []
            # for cols
            cindex_order2 = []
            for space in cspaces:
                sbR, sbC = self.space_map[space]
                obR, obC = other.space_map[space]
                assert sbC.N == obR.N
                nI *= sbC.N
                nR *= sbR.N
                nC *= obC.N
                cspace_map[space] = SpaceBasis(sbR, obC)
                cindex_order.append(SpaceIndex(space, col=False, N=sbR.N))
                cindex_order2.append(SpaceIndex(space, col=True, N=obC.N))
            snR = self.nR // nR
            snC = self.nC // nI
            onR = other.nR // nI
            onC = other.nC // nC

            A = self._matrix.reshape(snR, snC, 1, 1, nR, nI)
            B = other._matrix.reshape(1, 1, onR, onC, nI, nC)
            C = A @ B
            n = self.__class__.__new__(self.__class__)
            n.nR = snR * onR * nR
            n.nC = snC * onC * nC
            n._matrix = C.reshape(n.nR, n.nC)
            # print('out_shape:', n._matrix.shape, n.nR, n.nC)

            new_space_map = dict(self.space_map)
            new_space_map.update(other.space_map)
            new_space_map.update(cspace_map)
            n.space_map = new_space_map

            n.space_segments = (
                self.space_segments[:-1] + other.space_segments[:-1] + [tuple(cspaces)]
            )
            n.index_order = (
                self.index_order[: len(self.index_order) - 2 * len(cspaces)]
                + other.index_order[: len(other.index_order) - 2 * len(cspaces)]
                + cindex_order
                + cindex_order2
            )
        else:
            A = self._matrix.reshape(self.nR, self.nC, 1, 1)
            B = other._matrix.reshape(1, 1, other.nR, other.nC)
            # outer product, like used in "join"
            C = A * B
            n = self.__class__.__new__(self.__class__)
            n.nR = self.nR * other.nR
            n.nC = self.nC * other.nC
            n._matrix = C.reshape(n.nR, n.nC)

            new_space_map = dict(self.space_map)
            new_space_map.update(other.space_map)
            n.space_map = new_space_map

            n.space_segments = self.space_segments + other.space_segments
            n.index_order = self.index_order + other.index_order
        return n

    @property
    def T(self):
        n = self.__class__.__new__(self.__class__)
        # have to ensure a transposable order
        self.segments_condense()
        n._matrix = self._matrix.T
        n.nR = self.nC
        n.nC = self.nR
        n.space_map = {
            space: SpaceBasis(sb.bC, sb.bR) for space, sb in self.space_map.items()
        }
        n.space_segments = list(self.space_segments)

        new_index_order = []
        for segment in n.space_segments:
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=False, N=n.space_map[space].bR.N)
                )
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=True, N=n.space_map[space].bC.N)
                )
        n.index_order = new_index_order
        return n

    @property
    def C(self):
        n = self.__class__.__new__(self.__class__)
        n._matrix = self._matrix.conj()
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    @property
    def A(self):
        n = self.__class__.__new__(self.__class__)
        # have to ensure a transposable order
        self.segments_condense()
        n._matrix = self._matrix.T.conj()
        n.nR = self.nC
        n.nC = self.nR
        n.space_map = {
            space: SpaceBasis(sb.bC, sb.bR) for space, sb in self.space_map.items()
        }
        n.space_segments = list(self.space_segments)

        new_index_order = []
        for segment in n.space_segments:
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=False, N=n.space_map[space].bR.N)
                )
            for space in segment:
                new_index_order.append(
                    SpaceIndex(space=space, col=True, N=n.space_map[space].bC.N)
                )
        n.index_order = new_index_order
        return n

    def __eq__(self, other):
        if other.space_map.keys() != self.space_map.keys():
            return False
        # TODO, check the basis
        other.segments_set(self.space_segments)
        if np.all(self._matrix == other._matrix):
            return True
        return False

    def __add__(self, other):
        s_d = {s: rc for s, rc in other.space_map.items() if s not in self.space_map}
        if s_d:
            self = self.copy()
            self.join_id(s_d)

        o_d = {s: rc for s, rc in self.space_map.items() if s not in other.space_map}
        if o_d:
            other = other.copy()
            other.join_id(o_d)

        assert(other.space_map.keys() == self.space_map.keys())
        other.segments_set(self.space_segments)

        n = self.__class__.__new__(self.__class__)
        n._matrix = self._matrix + other._matrix
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def __sub__(self, other):
        s_d = {s: rc for s, rc in other.space_map.items() if s not in self.space_map}
        if s_d:
            self = self.copy()
            self.join_id(s_d)

        o_d = {s: rc for s, rc in self.space_map.items() if s not in other.space_map}
        if o_d:
            other = other.copy()
            other.join_id(o_d)

        assert(other.space_map.keys() == self.space_map.keys())
        other.segments_set(self.space_segments)

        n = self.__class__.__new__(self.__class__)
        # TODO, check typing
        n._matrix = self._matrix - other._matrix
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def __rmul__(self, o_scalar):
        if not isinstance(o_scalar, numbers.Number):
            return NotImplemented
        n = self.__class__.__new__(self.__class__)
        n._matrix = o_scalar * self._matrix
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def __mul__(self, o_scalar):
        if not isinstance(o_scalar, numbers.Number):
            return NotImplemented
        n = self.__class__.__new__(self.__class__)
        n._matrix = self._matrix * o_scalar
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def __truediv__(self, o_scalar):
        if not isinstance(o_scalar, numbers.Number):
            return NotImplemented
        n = self.__class__.__new__(self.__class__)
        n._matrix = self._matrix / o_scalar
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def __pos__(self):
        return self

    def __neg__(self):
        n = self.__class__.__new__(self.__class__)
        n._matrix = -self._matrix
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

    def expm(self, exp=None):
        n = self.__class__.__new__(self.__class__)
        # TODO, could check row/col basis match
        self.segments_condense()
        if exp is not None:
            n._matrix = expm(exp * self._matrix)
        else:
            n._matrix = expm(self._matrix)
        n.nR = self.nR
        n.nC = self.nC
        n.space_map = dict(self.space_map)
        n.index_order = list(self.index_order)
        n.space_segments = list(self.space_segments)
        return n

empty0 = Operator.bare(value=0)
empty1 = Operator.bare(value=1)

