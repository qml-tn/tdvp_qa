from jax.scipy.linalg import qr
from jax import jit
import jax.numpy as jnp
from jax import random
import numpy as np

from tdvp_qa.utils import annealing_energy_canonical, right_hamiltonian, left_hamiltonian, full_effective_hamiltonian_A


def mps_overlap(tensors1, tensors2):
    overlap = jnp.array([[1.]])
    n = len(tensors1)
    assert n == len(
        tensors2), "MPS should have the same length but have lengths {len(tensors1)} and {len(tensors2)}."
    for i in range(n):
        overlap = jnp.einsum("ij,ikl->ljk", overlap, tensors1[i])
        overlap = jnp.einsum("lkj,kjm->lm", overlap, jnp.conj(tensors2[i]))
    return overlap[0, 0]


def np_mps_overlap(tensors1, tensors2):
    overlap = jnp.array([[1.]])
    n = len(tensors1)
    assert n == len(
        tensors2), "MPS should have the same length but have lengths {len(tensors1)} and {len(tensors2)}."
    for i in range(n):
        overlap = np.einsum("ij,ikl->ljk", overlap, tensors1[i])
        overlap = np.einsum("lkj,kjm->lm", overlap, np.conj(tensors2[i]))
    return overlap[0, 0]


@jit
def _norm(nrm, A):
    nrm = jnp.einsum("ij,jkl->ikl", nrm, A)
    nrm = jnp.einsum("ikl,ikm->ml", nrm, jnp.conj(A))
    return nrm


RIGHT_CANONICAL = "right canonical"
LEFT_CANONICAL = "left canonical"
CENTRAL_CANONICAL = "central canonical"
NOT_ORDERED = "unordered"


class MPS_C():
    def __init__(self, tensors, centers=[], key=42):
        # The expected tensor shape is n, D, d, D
        n = len(tensors)
        self.n = n
        self.d = tensors[0].shape[1]
        self.tensors = [jnp.array(A.copy()) for A in tensors]
        self.order = NOT_ORDERED
        self.centers = centers
        if len(centers) > 0:
            self.order = CENTRAL_CANONICAL

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
        if self.order == CENTRAL_CANONICAL:
            sl = self.centers[i]
            sr = self.centers[i+1]
            A = self.tensors[i]
            A = jnp.einsum("i,ijk,k->ijk", sl, A, sr)
        return self.tensors[i]

    def copy_tensors(self):
        tensors = [A.copy() for A in self.tensors]
        return tensors

    def copy_centers(self):
        centers = [C.copy() for C in self.centers]
        return centers

    def set_tensors(self, tensors, copy=False):
        if copy:
            self.tensors = [A.copy() for A in tensors]
        else:
            self.tensors = tensors

    def set_centers(self, centers, copy=False):
        if copy:
            self.centers = [C.copy() for C in centers]
        else:
            self.centers = centers

    def set_tensor(self, i, A):
        # assert i<self.n, f"Can not access tensor {i}. Size of the MPS is {self.n}"
        self.tensors[i] = A

    def set_center(self, i, C):
        # assert i<len(self.centers)-1, f"Can not access center {i}. Size of the centers is {len(self.centers)}"
        self.centers[i] = C

    def move_right(self, i):
        assert self.order != CENTRAL_CANONICAL, "The state should not be in the central canonical form"
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
        assert self.order != CENTRAL_CANONICAL, "The state should not be in the central canonical form"
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

    def central_canonical(self):
        n = self.n
        self.right_canonical()
        sinv = jnp.array([[1.]])
        eps = 1e-10
        r = jnp.array([[1.0]])
        self.centers = [jnp.array([[1.0]])]
        for i in range(n-1):
            Ai = self.get_tensor(i)
            Ai = jnp.einsum("ij,jkl->ikl", r, Ai)
            dims = Ai.shape
            D = dims[-1]
            u, s, v = jnp.linalg.svd(Ai, [-1, dims[-1]], mode='full')
            u = u[:, :D]
            s = s[:D]
            v = v[:D]
            self.centers.append(s)
            Ai = jnp.einsum("ij,jkl->ikl", sinv, u)
            self.tensors[i] = Ai.reshape(dims[:-1]+[-1])
            sinv = 1.0/(s+eps)
            r = jnp.einsum("i,ij->ij", s, v)
        Ai = self.get_tensor(n-1)
        Ai = jnp.einsum("ij,jkl->ikl", v, Ai)
        self.tensors[n-1] = Ai
        self.centers.append(jnp.array([[1.0]]))
        self.order = CENTRAL_CANONICAL

    def right_canonical(self):
        if self.order == CENTRAL_CANONICAL:
            
        else:
            n = self.n
            for i in range(n-1, 0, -1):
                self.move_left(i)
        self.centers = []
        self.order = RIGHT_CANONICAL

    def left_canonical(self):
        assert self.order != CENTRAL_CANONICAL, f"The state should not be in the cnetral canonical form"
        n = self.n
        for i in range(n-1):
            self.move_right(i)
        self.centers = []
        self.order = LEFT_CANONICAL

    def construct_state(self):
        if self.order == CENTRAL_CANONICAL:
            raise ValueError(
                f"The state should not be in the central canonical form!")
        assert self.n <= 12, f"Can not construct a state with more than 12 spins. The state has {self.n} spins"
        psi = self.get_tensor(0)
        for i in range(1, self.n):
            psi = jnp.einsum("...i,ijk", psi, self.get_tensor(i))
        return jnp.reshape(psi, [-1])

    def sample(self):
        if self.order != RIGHT_CANONICAL:
            raise ValueError(
                f"The state should be in the right canonical form bit is {self.order}!")
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
        if self.order == CENTRAL_CANONICAL:
            raise ValueError(
                f"The state should not be in the central canonical form!")
        if mps2.order == CENTRAL_CANONICAL:
            raise ValueError(
                f"The second state should not be in the central canonical form!")
        return mps_overlap(self.tensors, mps2.tensors)

    def copy(self):
        return MPS_C(self.tensors, centers=self.centers)
