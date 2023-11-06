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
        # We assume that the state is in the right canonical form
        n = self.n
        r = jnp.array([[1]])
        sample = np.zeros(n)
        key, subkey = random.split(self.key)
        for i in range(n-1):
            A = self.tensors[i]
            A = jnp.einsum("ij,jkl->ikl", r, A)

            p0 = jnp.linalg.norm(A[:, 0, :])**2
            p1 = jnp.linalg.norm(A[:, 1, :])**2

            assert np.isclose(
                p0+p1, 1), f"Distribution is not normalized p0={p0}, p1={p1}, p0+p1={p0+p1}."

            r = random.uniform(subkey)
            if r < p1:
                sample[i] = 1
                A = A[:, 1, :]
            else:
                A = A[:, 0, :]

            _, r = qr(A)
            r = r/jnp.linalg.norm(r)

            key, subkey = random.split(key)
        self.key = key
        return sample

    def overlap(self, mps2):
        return mps_overlap(self.tensors, mps2.tensors)
