from jax import jit
import jax.numpy as jnp
from tdvp_qa.mps import MPS
from jax import random
from jax.scipy.linalg import svd, expm
from tqdm import tqdm
from GracefulKiller import GracefulKiller
import time

import numpy as np


@jit
def right_hamiltonian(A, Hr0, H0):
    Hr = jnp.einsum("aiu,umd->aimd", A, Hr0)
    Hr = jnp.einsum("njim,aimd->anjd", H0, Hr)
    Hr = jnp.einsum("bjd,anjd->anb", jnp.conj(A), Hr)
    return Hr


@jit
def left_hamiltonian(A, Hl0, H0):
    Hl = jnp.einsum("uia,umd->aimd", A, Hl0)
    Hl = jnp.einsum("mjin,aimd->anjd", H0, Hl)
    Hl = jnp.einsum("djb,anjd->anb", jnp.conj(A), Hl)
    return Hl


def right_context(mps: type[MPS], mpo):
    # Here we assume that the mps is already in the right canonical form
    n = mps.n
    Hright = [jnp.array([[[1.]]])]
    for i in range(n-1, 0, -1):
        H0 = mpo[i]
        A = mps.get_tensor(i)
        Hr = right_hamiltonian(A, Hright[0], H0)
        Hright = [Hr] + Hright
    return Hright


@jit
def effective_hamiltonian_A(Hl, Hr, H0):
    Heff = jnp.einsum("umd,mijn->diujn", Hl, H0)
    Heff = jnp.einsum("diujn,anb->dibuja", Heff, Hr)
    return Heff


@jit
def effective_hamiltonian_C(Hl, Hr):
    Heff = jnp.einsum("umd,amb->dbua", Hl, Hr)
    return Heff


class TDVP_QA():
    def __init__(self, mpo0, mpo1, tensors, slope, dt, max_slope=0.1, min_slope=1e-5, adaptive=False, compute_states=False, stochastic=False, key=42):
        # mpo0, mpo1 are simple nxMxdxdxM tensors containing the MPO representations of H0 and H1
        self.mpo0 = [jnp.array(A) for A in mpo0]
        self.mpo1 = [jnp.array(A) for A in mpo1]
        # The MPS at initialization should be in the right canonical form
        self.mps = MPS(tensors, key)
        self.mps.right_canonical()

        self.entropy = 0

        self.n = self.mps.n
        self.d = self.mps.d

        self.lamb = slope/2.
        self.max_slope = max_slope
        self.min_slope = min_slope
        self.slope = slope
        self.dt = dt
        self.adaptive = adaptive
        self.stochastic = stochastic
        self.compute_states = compute_states

        self.Hright0 = right_context(self.mps, self.mpo0)
        self.Hright1 = right_context(self.mps, self.mpo1)
        self.Hleft0 = None
        self.Hleft1 = None

        self.key = random.PRNGKey(key)

        self.tstart = time.time()
        self.killer = GracefulKiller()

    def get_dt(self):
        if not self.stochastic:
            return self.dt/2
        key, subkey = random.split(self.key)
        dtr = jnp.real(self.dt/2)
        dti = jnp.imag(self.dt/2) * random.uniform(subkey)
        dt = dtr + 1j*dti
        self.key = key
        return dt

    def update_lambda(self):
        self.lamb = np.min([1, self.lamb + self.slope])

    def get_couplings(self):
        a = np.max([1 - self.lamb, 0.])
        b = np.min([self.lamb, 1])
        return -a, b

    def calculate_energy(self):
        Hl0 = jnp.array([[[1.]]])
        Hl1 = jnp.array([[[1.]]])
        Hr0 = self.Hright0[0]
        Hr1 = self.Hright1[0]
        H0 = self.mpo0[0]
        H1 = self.mpo1[0]

        A = self.mps.get_tensor(0)
        Dl, d, Dr = A.shape

        a, b = self.get_couplings()

        # Effective Hamiltonian for A
        dd = Dl*d*Dr
        Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        Ha = a * Ha0 + b * Ha1

        A = jnp.reshape(A, [-1])
        return jnp.einsum("i,ij,j", jnp.conj(A), Ha, A)

    def right_sweep(self, dt):
        Hleft0 = [jnp.array([[[1.]]])]
        Hleft1 = [jnp.array([[[1.]]])]
        # Assumes that the Hright is already prepared and that the state is in the right canonical form

        n = self.n

        a, b = self.get_couplings()
        for i in range(n-1):
            Hl0 = Hleft0[i]
            Hl1 = Hleft1[i]
            Hr0 = self.Hright0[i]
            Hr1 = self.Hright1[i]
            H0 = self.mpo0[i]
            H1 = self.mpo1[i]

            A = self.mps.get_tensor(i)
            Dl, d, Dr = A.shape

            # Effective Hamiltonian for A
            dd = Dl*d*Dr
            Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
            Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
            Ha = a * Ha0 + b * Ha1

            # Updating A
            A = jnp.reshape(A, [dd])
            eA = expm(-1j*dt*Ha)
            A = eA @ A
            A = jnp.reshape(A, [Dl, d, Dr])
            A = A/jnp.linalg.norm(A)
            self.mps.set_tensor(i, A)
            r, Ar = self.mps.move_right(i)

            # Calculating new Hleft
            A = self.mps.get_tensor(i)
            Hl0 = left_hamiltonian(A, Hleft0[i], H0)
            Hl1 = left_hamiltonian(A, Hleft1[i], H1)
            Hleft0.append(Hl0)
            Hleft1.append(Hl1)

            # Effective Hamiltonian for C
            Dl, Dr = r.shape
            Hc0 = jnp.reshape(effective_hamiltonian_C(
                Hl0, Hr0), [Dl*Dr, Dl*Dr])
            Hc1 = jnp.reshape(effective_hamiltonian_C(
                Hl1, Hr1), [Dl*Dr, Dl*Dr])
            Hc = a * Hc0 + b * Hc1

            # Updating C
            C = jnp.reshape(r, [Dl*Dr])
            eC = expm(1j*dt*Hc)
            C = eC @ C
            C = jnp.reshape(C, [Dl, Dr])
            C = C/jnp.linalg.norm(C)
            Ar = jnp.einsum("ij,jkl->ikl", C, Ar)
            self.mps.set_tensor(i+1, Ar)

        #  Handling the last site update

        i = n-1
        Hl0 = Hleft0[i]
        Hl1 = Hleft1[i]
        Hr0 = self.Hright0[i]
        Hr1 = self.Hright1[i]
        H0 = self.mpo0[i]
        H1 = self.mpo1[i]

        A = self.mps.get_tensor(i)
        Dl, d, Dr = A.shape

        # Effective Hamiltonian for A
        dd = Dl*d*Dr
        Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        Ha = a * Ha0 + b * Ha1

        # Updating A
        A = jnp.reshape(A, [dd])
        eA = expm(-1j*dt*Ha)
        A = eA @ A
        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mps.set_tensor(i, A)

        # Setting left effective Hamiltonians
        self.Hleft0 = Hleft0
        self.Hleft1 = Hleft1

    def left_sweep(self, dt):
        Hright0 = [jnp.array([[[1.]]])]
        Hright1 = [jnp.array([[[1.]]])]
        # Assumes that the Hleft is already prepared and that the state is in the left canonical form

        n = self.n

        a, b = self.get_couplings()
        for i in range(n-1, 0, -1):
            Hl0 = self.Hleft0[i]
            Hl1 = self.Hleft1[i]
            Hr0 = Hright0[0]
            Hr1 = Hright1[0]
            H0 = self.mpo0[i]
            H1 = self.mpo1[i]

            A = self.mps.get_tensor(i)
            Dl, d, Dr = A.shape

            # Effective Hamiltonian for A
            dd = Dl*d*Dr
            Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
            Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
            Ha = a * Ha0 + b * Ha1

            # Updating A
            A = jnp.reshape(A, [dd])
            eA = expm(-1j*dt*Ha)
            A = eA @ A
            A = jnp.reshape(A, [Dl, d, Dr])
            A = A/jnp.linalg.norm(A)
            self.mps.set_tensor(i, A)
            Al, r = self.mps.move_left(i)

            # Calculating new Hright
            A = self.mps.get_tensor(i)
            Hr0 = right_hamiltonian(A, Hr0, H0)
            Hr1 = right_hamiltonian(A, Hr1, H1)
            Hright0 = [Hr0] + Hright0
            Hright1 = [Hr1] + Hright1

            # Effective Hamiltonian for C
            Dl, Dr = r.shape
            Hc0 = jnp.reshape(effective_hamiltonian_C(
                Hl0, Hr0), [Dl*Dr, Dl*Dr])
            Hc1 = jnp.reshape(effective_hamiltonian_C(
                Hl1, Hr1), [Dl*Dr, Dl*Dr])
            Hc = a * Hc0 + b * Hc1

            # Updating C
            C = jnp.reshape(r, [Dl*Dr])
            eC = expm(1j*dt*Hc)
            C = eC @ C
            C = jnp.reshape(C, [Dl, Dr])
            C = C/jnp.linalg.norm(C)
            Al = jnp.einsum("ijk,kl->ijl", Al, C)

            # Calculating the entropy on the fly
            if i == n//2:
                _, s, _ = svd(C)
                self.entropy = -np.log(s) @ s

            self.mps.set_tensor(i-1, Al)

        # Updating the first site
        i = 0
        Hl0 = self.Hleft0[i]
        Hl1 = self.Hleft0[i]
        Hr0 = Hright0[0]
        Hr1 = Hright1[0]
        H0 = self.mpo0[i]
        H1 = self.mpo1[i]

        A = self.mps.get_tensor(i)
        Dl, d, Dr = A.shape

        # Effective Hamiltonian for A
        dd = Dl*d*Dr
        Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        Ha = a * Ha0 + b * Ha1

        # Updating A
        A = jnp.reshape(A, [dd])
        eA = expm(-1j*dt*Ha)
        A = eA @ A
        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mps.set_tensor(i, A)

        # Setting right effective Hamiltonians
        self.Hright0 = Hright0
        self.Hright1 = Hright1

    def update_tdvp_state(self, tensors, lamb, slope):
        self.mps.set_tensors(tensors)
        self.lamb = lamb
        self.slope = slope

    def evolve(self, evolve_final=False, data=None):
        if data is None:
            energies = []
            energiesr = []
            entropies = []
            slopes = []
            states = []
        else:
            energies = data["energy"]
            energiesr = data["energyr"]
            entropies = data["entropy"]
            slopes = data["slope"]
            states = data["state"]

        pbar = tqdm(total=1, position=0, leave=True)
        pbar.update(self.lamb)
        while (self.lamb < 1):
            dt = self.get_dt()
            er = 0
            if self.adaptive:
                # backup state
                tensors = self.mps.copy_tensors()

                # real update
                dtr = np.real(dt)
                self.right_sweep(dtr)
                self.left_sweep(dtr)
                er = self.calculate_energy()

                # Revert to backup
                self.mps.set_tensors(tensors, copy=True)

            # complex update
            self.right_sweep(dt)
            self.left_sweep(dt)
            ec = self.calculate_energy()

            if self.compute_states:
                states.append(self.mps.construct_state())

            increase_lambda = True
            if self.adaptive:
                # correct the slope
                if self.slope > self.min_slope and abs(er-ec) > 0.01:
                    self.slope = np.max([self.slope/2., self.min_slope])
                    self.mps.set_tensors(tensors)
                    increase_lambda = False
                if abs(er-ec) < 0.001:
                    self.slope = np.min([self.slope*2., self.max_slope])

            if increase_lambda:
                pbar.update(self.slope)
                self.update_lambda()
                energies.append(ec)
                entropies.append(self.entropy/np.log(2.0))
                slopes.append(self.slope)
                if self.adaptive:
                    energiesr.append(er)

            tcurrent = time.time()
            if tcurrent-self.tstart > 3600*47 or self.killer.kill_now:
                print(
                    f"Killing program after {int(tcurrent-self.tstart)} seconds.")
                break

        pbar.close()

        if evolve_final and not self.killer.kill_now:
            dt = self.get_dt()
            er = 0
            if self.adaptive:
                # backup state
                tensors = self.mps.copy_tensors()

                # real update
                dtr = np.real(dt)
                self.right_sweep(dtr)
                self.left_sweep(dtr)
                er = self.calculate_energy()

                # Revert to backup
                self.mps.set_tensors(tensors, copy=True)

            # complex update
            self.right_sweep(dt)
            self.left_sweep(dt)
            ec = self.calculate_energy()

            if self.compute_states:
                states.append(self.mps.construct_state())

            energies.append(ec)
            entropies.append(self.entropy/np.log(2.0))
            slopes.append(self.slope)
            if self.adaptive:
                energiesr.append(er)

        data = {"energy": energies, "energyr": energiesr,
                "entropy": entropies, "slope": slopes, "state": states}
        return data
