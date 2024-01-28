from jax import jit
import jax.numpy as jnp
from jax import random, grad
from jax.scipy.linalg import svd
from tqdm import tqdm
from GracefulKiller import GracefulKiller
import time
import numpy as np


from tdvp_qa.mps import MPS
from tdvp_qa.utils import annealing_energy_canonical, right_hamiltonian, left_hamiltonian, right_context, effective_hamiltonian_A, effective_hamiltonian_C


class TDVP_QA_AD():
    def __init__(self, mpo0, mpo1, tensors, slope, dt, lamb=0, max_slope=0.05, min_slope=1e-6, adaptive=False, compute_states=False, key=42, slope_omega=1e-3, ds=0.01, scale_gap=False):
        # mpo0, mpo1 are simple nxMxdxdxM tensors containing the MPO representations of H0 and H1
        self.mpo0 = [jnp.array(A) for A in mpo0]
        self.mpo1 = [jnp.array(A) for A in mpo1]
        # The MPS at initialization should be in the right canonical form
        self.mps = MPS(tensors, key)
        self.mps.right_canonical()

        self.entropy = 0

        self.n = self.mps.n
        self.d = self.mps.d
        self.omega0 = 1
        self.scale_gap = scale_gap

        self.lamb = lamb
        self.max_slope = max_slope
        self.min_slope = min_slope
        self.slope = slope
        self.slope_omega = slope_omega
        self.dt = dt
        self.adaptive = adaptive
        self.compute_states = compute_states
        self.ds = ds

        self.Hright0 = right_context(self.mps, self.mpo0)
        self.Hright1 = right_context(self.mps, self.mpo1)
        self.Hleft0 = None
        self.Hleft1 = None

        self.key = random.PRNGKey(key)

        self.tstart = time.time()
        self.killer = GracefulKiller()

    def get_dt(self):
        key, subkey = random.split(self.key)
        dtr = jnp.real(self.dt)
        dti = jnp.imag(self.dt) * random.uniform(subkey)
        dt = dtr + 1j*dti
        self.key = key
        return dt

    def update_lambda(self):
        self.lamb = np.clip(self.lamb + self.slope, 0, 1)

    def get_couplings(self, lamb=None):
        if lamb is None:
            lamb = self.lamb
        a = np.max([1 - lamb, 0.])
        b = np.min([lamb, 1.])
        return -a, b

    def energy_mpo(self, mpo, mps):
        e = jnp.array([[[1.0]]])
        n = self.n
        nrm = jnp.array([[1.0]])
        for i in range(n):
            A = mps[i]
            O = mpo[i]
            e = jnp.einsum("umd,uiU->Umid", e, A)
            e = jnp.einsum("Umid,djD->UmijD", e, jnp.conj(A))
            e = jnp.einsum("UmijD,mijM->UMD", e, O)

            nrm = jnp.einsum("ud,uiU->Uid", nrm, A)
            nrm = jnp.einsum("Uid,diD->UD", nrm, jnp.conj(A))

        e = jnp.real(e[0, 0, 0])
        nrm = jnp.real(nrm[0, 0])
        return e/nrm

    def energy_right_canonical(self, lamb=None):
        Hl0 = jnp.array([[[1.]]])
        Hl1 = jnp.array([[[1.]]])
        Hr0 = self.Hright0[0]
        Hr1 = self.Hright1[0]
        H0 = self.mpo0[0]
        H1 = self.mpo1[0]

        A = self.mps.get_tensor(0)
        return annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, lamb, A)

    def energy_left_canonical(self, lamb=None):
        Hr0 = jnp.array([[[1.]]])
        Hr1 = jnp.array([[[1.]]])
        Hl0 = self.Hleft0[-1]
        Hl1 = self.Hleft1[-1]
        H0 = self.mpo0[-1]
        H1 = self.mpo1[-1]

        n = self.n
        A = self.mps.get_tensor(n-1)
        # Dl, d, Dr = A.shape

        # a, b = self.get_couplings(lamb)

        # # Effective Hamiltonian for A
        # dd = Dl*d*Dr
        # Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        # Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        # Ha = a * Ha0 + b * Ha1

        # A = jnp.reshape(A, [-1])
        # return jnp.einsum("i,ij,j", jnp.conj(A), Ha, A)
        return annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, lamb, A)

    def evolve_with_local_H(self, A, H, dt, omega0, omega_scale):
        val, vec = jnp.linalg.eigh(H)
        if len(val) <= 1:
            return A, omega0, omega_scale

        # calculate gaps
        gap = val[1]-val[0]
        # mean_gap = np.mean(np.diff(val))
        omega0 = np.min([omega0, gap])
        if self.scale_gap:
            # val = (val-val[0])/self.omega0
            # If the gap is 0 we have a problem here
            val = (val-val[0])/(gap + 1e-32)
        omega_scale = np.min([omega_scale, val[1]-val[0]])
        # Evolve for a time dt
        if self.scale_gap and abs(np.imag(dt)) >= 1.0:
            # if gap < 1e-2:
            #     print("small gap",gap/mean_gap,gap,mean_gap)
            #     return (vec[:, 0]+vec[:, 1])/np.sqrt(2.), omega0, omega_scale
            return vec[:, 0],  omega0, omega_scale

        A = jnp.einsum("ji,j->i", jnp.conj(vec), A)
        A = jnp.einsum("i,i->i", jnp.exp(-1j*val*dt), A)
        A = jnp.einsum("ij,j->i", vec, A)
        return A, omega0, omega_scale

    def right_sweep(self, dt, lamb=None):
        Hleft0 = [jnp.array([[[1.]]])]
        Hleft1 = [jnp.array([[[1.]]])]
        # Assumes that the Hright is already prepared and that the state is in the right canonical form

        a, b = self.get_couplings(lamb)
        omega0 = jnp.inf
        omega_scale = jnp.inf

        n = self.n

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
            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, dt, omega0, omega_scale)

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

            C = jnp.reshape(r, [Dl*Dr])
            Hc = a * Hc0 + b * Hc1

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, -np.conj(dt), omega0, omega_scale)

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
        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, dt, omega0, omega_scale)

        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mps.set_tensor(i, A)

        # Setting left effective Hamiltonians
        self.Hleft0 = Hleft0
        self.Hleft1 = Hleft1

        return omega0, omega_scale

    def left_sweep(self, dt, lamb=None):
        Hright0 = [jnp.array([[[1.]]])]
        Hright1 = [jnp.array([[[1.]]])]
        # Assumes that the Hleft is already prepared and that the state is in the left canonical form

        n = self.n
        a, b = self.get_couplings(lamb)
        omega0 = jnp.inf
        omega_scale = jnp.inf

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
            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, dt, omega0, omega_scale)

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
            C = jnp.reshape(r, [Dl*Dr])
            Hc = a * Hc0 + b * Hc1

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, -np.conj(dt), omega0, omega_scale)

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
        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, dt, omega0, omega_scale)

        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mps.set_tensor(i, A)

        # Setting right effective Hamiltonians
        self.Hright0 = Hright0
        self.Hright1 = Hright1

        return omega0, omega_scale

    def right_left_sweep(self, dt, lamb=None):
        omega0r, omega_scaler = self.right_sweep(dt/2., lamb)
        omega0l, omega_scalel = self.left_sweep(dt/2., lamb)
        return np.min([omega0l, omega0r]), np.min([omega_scalel, omega_scaler, 1.0])

    def apply_gradients(self, gradients, lr=1e-3):
        n = self.n
        for i in range(n):
            A = self.mps.get_tensor(i)
            A = A - lr*gradients[i]
            self.mps.set_tensor(i,A)
        self.mps.normalize()

    def evolve(self, data=None, auto_grad=False):
        keys = ["energy", "omega0", "entropy",
                "slope", "state", "var_gs", "s", "ds_overlap", "init_overlap"]
        if data is None:
            data = {}
            for key in keys:
                data[key] = []
        else:
            self.omega0 = data["omega0"][-1]

        for key in keys:
            dkeys = list(data.keys())
            if key not in dkeys:
                data[key] = []

        pbar = tqdm(total=1, position=0, leave=True)
        pbar.update(self.lamb)
        if self.lamb == 0:
            k = 1
        else:
            k = int(np.ceil(self.lamb/self.ds))

        mps_prev = self.mps.copy()
        mps0 = self.mps.copy()

        def energy(mps):
            e0 = self.energy_mpo(self.mpo0, mps)
            e1 = self.energy_mpo(self.mpo1, mps)
            a = jnp.clip(1 - self.lamb, 0, 1)
            b = jnp.clip(self.lamb, 0, 1)
            e = a*e0+b*e1
            return e

        energy_gradient = jit(grad(energy))

        omega0 = 1

        while (self.lamb < 1):
            dt = self.get_dt()
            # full step update right
            # lamb = np.clip(self.lamb + self.slope, 0, 1)
            self.update_lambda()
            lamb = self.lamb
            if not auto_grad:
                omega0, _ = self.right_left_sweep(dt, lamb)
                ec_prev = self.energy_right_canonical(lamb)
                for _ in range(100):
                    omega0, _ = self.right_left_sweep(dt, lamb)
                    ec = self.energy_right_canonical(lamb)
                    print(ec-ec_prev)
                    if abs(ec-ec_prev)<1e-6:
                        break
                    ec_prev = ec
            else:
                mglist = []
                for gstep in range(500):
                    gradients = energy_gradient(self.mps.tensors)
                    mg = np.max([jnp.linalg.norm(g) for g in gradients])
                    mglist.append(mg)
                    print(mg)
                    self.apply_gradients(gradients,lr=abs(dt))
                    if mg<1e-4:
                        print(f"gstep {gstep}")
                        break
                
                # self.mps.right_canonical()
                ec = energy(self.mps.tensors)
                # print(mglist[0],mglist[-1],jnp.linalg.norm(gradients[0]))
                # print(mglist)

            self.omega0 = omega0

            if self.adaptive:
                self.slope = np.clip(
                    omega0*self.slope_omega, self.min_slope, self.max_slope)

            data["omega0"].append(float(np.real(omega0)))

            if lamb >= k*self.ds:
                data["energy"].append(float(np.real(ec)))
                data["entropy"].append(
                    float(np.real(self.entropy/np.log(2.0))))
                data["s"].append(lamb)
                data["ds_overlap"].append(abs(self.mps.overlap(mps_prev)))
                data["init_overlap"].append(abs(self.mps.overlap(mps0)))
                mps_prev = self.mps.copy()
                k = k+1
                if self.compute_states:
                    dmrg_mps = self.mps.copy()
                    dmrg_mps.dmrg(lamb, self.mpo0, self.mpo1,
                                  self.Hright0, self.Hright1, sweeps=20)
                    # data["var_gs"].append(np.array(dmrg_mps.construct_state()))
                    data["state"].append([np.array(A)
                                         for A in self.mps.tensors])
                    data["var_gs"].append([np.array(A)
                                          for A in dmrg_mps.tensors])
                    # print(lamb, ec, self.slope, self.omega0, omega_scale)
            data["slope"].append(float(np.real(self.slope)))
            pbar.update(float(np.real(self.slope)))

            tcurrent = time.time()
            if tcurrent-self.tstart > 3600*47 or self.killer.kill_now:
                print(
                    f"Killing program after {int(tcurrent-self.tstart)} seconds.")
                break
        data["mps"] = [np.array(A) for A in self.mps.tensors]
        pbar.close()
        return data
