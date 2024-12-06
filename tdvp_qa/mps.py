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
        overlap = np.einsum("ij,ikl->ljk", overlap, np.conj(tensors1[i]))
        overlap = np.einsum("lkj,kjm->lm", overlap, tensors2[i])
    return overlap[0, 0]

############# NOT WORKING START ##############
# def np_mps_norm(tensors):
#     n = len(tensors)
#     nrm = np.array([[1.]])
#     for i in range(n):
#         A = tensors[i]
#         nrm = np.einsum("ij,jkl->ikl", nrm, A)
#         nrm = np.einsum("ikl,ikm->ml", nrm, np.conj(A))
#     return np.sqrt(nrm[0, 0])

# def np_mps_difference(tensors1, tensors2, calc_norm=True):
#     # We assume that the norm of the MPS2 is 1.
#     n = len(tensors1)
#     tensors =  []
#     assert n == len(
#         tensors2), "MPS should have the same length but have lengths {len(tensors1)} and {len(tensors2)}."
#     for i in range(n):
#         A1 = tensors1[i]
#         A2 = tensors2[i]
#         dims1 = np.array(A1.shape)
#         dims2 = np.array(A2.shape)
#         A = np.zeros(dims1+dims2)
#         A[:dims1[0],:dims1[1],:dims1[2]] = A1
#         A[dims1[0]:,dims1[1]:,dims1[2]:] = A2

#     overlap = np_mps_overlap(tensors1,tensors2)

#     el = np.array([[1,-overlap]])
#     A = np.einsum("ai,ijk->ajk",el, tensors[0])
#     tensors[0] = A

#     er = np.array([[1],[1]])
#     A = np.einsum("ijk,kb->ijb",er, tensors[-1])
#     tensors[-1] = A
#     if calc_norm:
#         return np_mps_norm(tensors)
#     return tensors
############# NOT WORKING END ##############


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
        self.dmax = np.max([A.shape[0] for A in tensors])
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

    def move_right(self, i, reduce_entropy=False, normalize=False):
        # assert i<self.n-1, f"Can not move site {i} to the right. Size of the MPS is {self.n}."
        Al = self.get_tensor(i)
        Ar = self.get_tensor(i+1)
        Dl, d, Dr = Al.shape
        dl = Dl*d
        q, r = qr(jnp.reshape(Al, [dl, Dr]), mode='full')
        Dr = np.min([q.shape[1], Dr])
        if normalize:
            r = r/jnp.linalg.norm(r)

        if self.dmax > 1 and reduce_entropy and (Dr >= self.dmax and dl > self.dmax):
            u, s, vt = jnp.linalg.svd(r, full_matrices=False)
            mins = jnp.min(s)
            if (mins > 1e-7):
                snew = np.zeros(len(s), dtype=np.float64)
                print(
                    f"Sampling singular values! min(s)={mins}, position={i}")
                # print(self.dmax, dl, Dl, d, Dr)
                cprobs = jnp.cumsum(s)/jnp.sum(s)
                key, subkey = random.split(self.key)
                rnd = random.uniform(subkey)
                self.key = key
                for j, cp in enumerate(cprobs):
                    if rnd < cp:
                        snew[j] = 1.0
                        break
                r = jnp.einsum("ij,j,jk->ik", u, snew, vt)

        Alnew = jnp.reshape(q[:, :Dr], [Dl, d, Dr])
        Arnew = jnp.einsum("ij,jkl->ikl", r[:Dr], Ar)
        self.set_tensor(i, Alnew)
        self.set_tensor(i+1, Arnew)
        return r[:Dr], Ar

    # @partial(jax.jit, static_argnums=0)
    def move_left(self, i, normalize=False):
        Al = self.get_tensor(i-1)
        Ar = self.get_tensor(i)
        Dl, d, Dr = Ar.shape
        dr = d*Dr
        qt, rt = qr(jnp.reshape(Ar, [Dl, dr]).T, mode='full')
        q = qt.T
        r = rt.T
        if normalize:
            r = r/jnp.linalg.norm(r)
        Dl = np.min([q.shape[0], Dl])
        Arnew = jnp.reshape(q[:Dl, :], [Dl, d, Dr])
        Alnew = jnp.einsum("ijk,kl->ijl", Al, r[:, :Dl])
        self.set_tensor(i-1, Alnew)
        self.set_tensor(i, Arnew)
        return Al, r[:, :Dl]

    def right_canonical(self, normalize=False):
        n = self.n
        for i in range(n-1, 0, -1):
            self.move_left(i, normalize=normalize)

    def left_canonical(self, normalize=False):
        n = self.n
        for i in range(n-1):
            self.move_right(i, normalize=normalize)

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

    def copy(self):
        return MPS(self.tensors)

    def dmrg(self, lamb, mpo0, mpo1, Hright0, Hright1, sweeps=20):
        n = self.n
        hright0 = [A.copy() for A in Hright0]
        hright1 = [A.copy() for A in Hright1]
        e0 = 1e5
        for nsweep in range(sweeps):
            # Starting the right sweep
            hleft0 = [jnp.array([[[1.]]])]
            hleft1 = [jnp.array([[[1.]]])]
            H0 = mpo0[0]
            Hl0 = hleft0[0]
            Hr0 = hright0[0]
            H1 = mpo1[0]
            Hl1 = hleft1[0]
            Hr1 = hright1[0]
            A = self.get_tensor(0)

            a = -max([1 - lamb, 0.])
            b = min([lamb, 1.])
            e1 = annealing_energy_canonical(
                Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A)
            # print(f"nsweep = {nsweep}. Err={e1-e0}")
            if np.abs(e1-e0) < 1e-10:
                # print(f"Converged in nsweep = {nsweep}. Err={e1-e0}")
                break
            e0 = e1
            for i in range(n-1):
                Dl, d, Dr = self.get_tensor(i).shape
                H0 = mpo0[i]
                Hl0 = hleft0[i]
                Hr0 = hright0[i]
                H1 = mpo1[i]
                Hl1 = hleft1[i]
                Hr1 = hright1[i]
                dd = Dl*d*Dr
                Ha = full_effective_hamiltonian_A(
                    Hl0, Hl1, H0, H1, Hr0, Hr1, lamb, dd)

                _, vec = jnp.linalg.eigh(Ha)
                Al = jnp.reshape(vec[:, 0], [Dl, d, Dr])

                self.set_tensor(i, Al)
                self.move_right(i)
                Alnew = self.get_tensor(i)

                Hl0 = left_hamiltonian(Alnew, Hl0, H0)
                Hl1 = left_hamiltonian(Alnew, Hl1, H1)
                hleft0.append(Hl0)
                hleft1.append(Hl1)

            # Starting the left sweep
            hright0 = [jnp.array([[[1.]]])]
            hright1 = [jnp.array([[[1.]]])]
            for i in np.arange(n-1, 0, -1):
                Dl, d, Dr = self.get_tensor(i).shape
                H0 = mpo0[i]
                Hl0 = hleft0[i]
                Hr0 = hright0[0]
                H1 = mpo1[i]
                Hl1 = hleft1[i]
                Hr1 = hright1[0]
                dd = Dl*d*Dr
                Ha = full_effective_hamiltonian_A(
                    Hl0, Hl1, H0, H1, Hr0, Hr1, lamb, dd)

                _, vec = jnp.linalg.eigh(Ha)
                Ar = jnp.reshape(vec[:, 0], [Dl, d, Dr])

                self.set_tensor(i, Ar)
                self.move_left(i)
                Arnew = self.get_tensor(i)

                Hr0 = right_hamiltonian(Arnew, Hr0, H0)
                Hr1 = right_hamiltonian(Arnew, Hr1, H1)
                hright0 = [Hr0] + hright0
                hright1 = [Hr1] + hright1


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
        B = np.zeros([dims[i], d, dims[i+1]], dtype=np.cdouble)
        if state[i] == 0:
            v = np.array([1., 1])/np.sqrt(2.)
        else:
            v = np.array([1., -1])/np.sqrt(2.)
        B[0, :, 0] = v
        mps.append(B)
    return mps


def initial_state_theta(n, Dmax, theta):
    '''
    List of MPS matrices for a particular initial product state

    Parameters:
      - n     : size of the system
      - Dmax  : maximum bond dimension
    '''
    d = 2
    mps = []
    dims = bond_dimensions(n, d, Dmax)
    for i in range(n):
        B = np.zeros([dims[i], d, dims[i+1]], dtype=np.cdouble)
        th = theta[i][0]
        fi = theta[i][1]
        v = np.array([np.cos(th/2), np.sin(th/2)*np.exp(-1j*fi)])
        B[0, :, 0] = v
        mps.append(B)
    return mps


def sample_tensors(mps, key=42):
    n = len(mps)
    key = random.PRNGKey(key)
    sample = np.zeros(n)
    Al = jnp.array([[1]])
    key, subkey = random.split(key)
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


def truncated_mps_tensors(mps: MPS, dmax=1):
    mps.right_canonical()
    tensors = mps.tensors
    n = len(tensors)
    M = mps.dmax
    for i in range(M-dmax):
        current_dmax = dmax
        for i in range(n-1):
            A = tensors[i]
            [Dl, d, Dr] = A.shape
            A = jnp.reshape(A, [Dl*d, -1])
            u, s, v = jnp.linalg.svd(A, full_matrices=False)
            dnew = np.max([dmax, Dr-1])
            dnew = np.min([dnew, len(s)])
            current_dmax = np.max([current_dmax, dnew])
            tensors[i] = jnp.reshape(u[:, :dnew], [Dl, d, -1])
            v = jnp.einsum("i,ij->ij", s[:dnew], v[:dnew])
            A = jnp.einsum("ia,ajk->ijk", v, tensors[i+1])
            tensors[i+1] = A

        for i in range(n-1, 0, -1):
            A = tensors[i]
            [Dl, d, Dr] = A.shape
            A = jnp.reshape(A, [-1, d*Dr])
            u, s, v = jnp.linalg.svd(A, full_matrices=False)
            dnew = np.max([dmax, Dl-1])
            dnew = np.min([dnew, len(s)])
            current_dmax = np.max([current_dmax, dnew])
            tensors[i] = jnp.reshape(v[:dnew], [-1, d, Dr])
            u = jnp.einsum("ij,j->ij", u[:, :dnew], s[:dnew])
            A = jnp.einsum("ijk,kb->ijb", tensors[i-1], u)
            tensors[i-1] = A
        if (current_dmax == dmax):
            break
    return tensors
