from jax.scipy.linalg import qr
from jax import jit
import jax.numpy as jnp
from jax import random

import numpy as np


def mps_overlap(tensors1, tensors2):
    overlap = jnp.array([[1.]])
    n = len(tensors1)
    assert n == len(
        tensors2), "MPS should have the same length but have lengths {len(tensors1)} and {len(tensors2)}."
    for i in range(n):
        overlap = jnp.einsum("ij,ikl->ljk", overlap, tensors1[i])
        overlap = jnp.einsum("lkj,kjm->lm", overlap, tensors2[i])
    return overlap[0, 0]


@jit
def _norm(nrm, A):
    nrm = jnp.einsum("ij,jkl->ikl", nrm, A)
    nrm = jnp.einsum("ikl,ikm->ml", nrm, jnp.conj(A))
    return nrm


class MPS():
    def __init__(self, tensors, key=42):
        # The expected tensor shape is n, D, d, D
        n = len(tensors)
        self.n = n
        self.d = tensors[0].shape[1]
        self.tensors = [jnp.array(A.copy()) for A in tensors]
        self.key = random.PRNGKey(key)

    def norm(self):
        n = self.n
        nrm = jnp.array([[1.]])
        for i in range(n):
            A = self.get_tensor(i)
            nrm = _norm(nrm, A)
        return jnp.sqrt(nrm[0, 0])

    def normalize(self):
        nrm = self.norm()
        A = self.get_tensor(0)/nrm
        self.set_tensor(0, A)

    def get_tensor(self, i):
        return self.tensors[i]

    def copy_tensors(self):
        tensors = [A.copy() for A in self.tensors]
        return tensors

    def set_tensors(self, tensors, copy=False):
        if copy:
            self.tensors = [A.copy() for A in tensors]
        else:
            self.tensors = tensors

    def set_tensor(self, i, A):
        # assert i<self.n, f"Can not access tensor {i}. Size of the MPS is {self.n}"
        self.tensors[i] = A

    def move_right(self, i):
        # assert i<self.n-1, f"Can not move site {i} to the right. Size of the MPS is {self.n}."
        Al = self.get_tensor(i)
        Ar = self.get_tensor(i+1)
        Dl, d, Dr = Al.shape
        dl = Dl*d
        q, r = qr(np.reshape(Al, [dl, Dr]), mode='full')
        Dr = np.min([q.shape[1], Dr])
        Alnew = jnp.reshape(q[:, :Dr], [Dl, d, Dr])
        Arnew = jnp.einsum("ij,jkl->ikl", r[:Dr], Ar)
        self.set_tensor(i, Alnew)
        self.set_tensor(i+1, Arnew)
        return r[:Dr], Ar

    # @partial(jax.jit, static_argnums=0)
    def move_left(self, i):
        Al = self.get_tensor(i-1)
        Ar = self.get_tensor(i)
        Dl, d, Dr = Ar.shape
        dr = d*Dr
        qt, rt = qr(np.reshape(Ar, [Dl, dr]).T, mode='full')
        q = qt.T
        r = rt.T
        Dl = np.min([q.shape[0], Dl])
        Arnew = jnp.reshape(q[:Dl, :], [Dl, d, Dr])
        Alnew = jnp.einsum("ijk,kl->ijl", Al, r[:, :Dl])
        self.set_tensor(i-1, Alnew)
        self.set_tensor(i, Arnew)
        return Al, r[:, :Dl]

    def right_canonical(self):
        n = self.n
        for i in range(n-1, 0, -1):
            self.move_left(i)

    def left_canonical(self):
        n = self.n
        for i in range(n-1):
            self.move_right(i)

    def construct_state(self):
        assert self.n <= 12, f"Can not construct a state with more than 12 spins. The state has {self.n} spins"
        psi = self.get_tensor(0)
        for i in range(1, self.n):
            psi = jnp.einsum("...i,ijk", psi, self.get_tensor(i))
        return jnp.reshape(psi, [-1])

    def sample(self):
        mps = self.tensors
        n = len(mps)
        sample = np.zeros(n)
        Al = jnp.array([[1]])
        self.key, subkey = random.split(self.key)
        rs = random.uniform(subkey, [n])
        for i in range(n):
            A = jnp.einsum("ij,jkl->ikl", Al, mps[i])
            p0 = jnp.linalg.norm(A[:, 0, :])**2
            p1 = jnp.linalg.norm(A[:, 1, :])**2

            assert np.isclose(
                p0+p1, 1), f"Distribution is not normalized p0={p0}, p1={p1}, p0+p1={p0+p1}."

            if rs[i] < p0:
                Al = A[:, 0, :]/jnp.sqrt(p0)
                sample[i] = 0
            else:
                sample[i] = 1
                Al = A[:, 1, :]/jnp.sqrt(p1)
        return sample

    def overlap(self, mps2):
        return mps_overlap(self.tensors, mps2.tensors)


def bond_dimensions(n, d, Dmax):
    '''
    List of bond dimensions

    Parameters:
      - n     : length of the system
      - d     : local Hilbert space size
      - Dmax  : maximum bond dimension
    '''
    dims = []
    nhalf = int((n+1)//2)
    for i in range(nhalf):
        dims.append(int(np.min([d**i, Dmax])))
    middle = []
    if np.mod(n, 2) == 0:
        middle = [int(np.min([d**nhalf, Dmax]))]
    return dims + middle + dims[::-1]


def initial_state(n, Dmax, K=0):
    '''
    List of MPS matrices for a particular initial product state

    Parameters:
      - n     : size of the system
      - Dmax  : maximum bond dimension
    '''
    d = 2
    mps = []
    state = np.array([int(a) for a in f"{K:b}".zfill(n)])
    dims = bond_dimensions(n, d, Dmax)
    for i in range(n):
        B = np.zeros([dims[i], d, dims[i+1]])
        if state[i]==0:
            v = np.array([1., 1])/np.sqrt(2.)
        else:
            v = np.array([1., -1])/np.sqrt(2.)
        B[0, :, 0] = v
        mps.append(B)
    return mps
