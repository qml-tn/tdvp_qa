from jax import jit, grad
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import svd
from tqdm import tqdm
from GracefulKiller import GracefulKiller
import time
import numpy as np


from tdvp_qa.mps import MPS
from tdvp_qa.utils import annealing_energy_canonical, right_hamiltonian, left_hamiltonian, right_context, effective_hamiltonian_A, effective_hamiltonian_C, linearised_specter


class TDVP_QA_V2():
    def __init__(self, mpo0, mpo1, tensors, slope, dt, lamb=0, max_slope=0.05, min_slope=1e-6, adaptive=False, compute_states=False, key=42, slope_omega=1e-3, ds=0.01, scale_gap=False, nitime=10, auto_grad=False, cyclic_path=False):
        # mpo0, mpo1 are simple nxMxdxdxM tensors containing the MPO representations of H0 and H1
        self.mpo0 = [jnp.array(A) for A in mpo0]
        self.mpo1 = [jnp.array(A) for A in mpo1]
        # The MPS at initialization should be in the right canonical form
        self.mps = MPS(tensors, key)
        self.mps.right_canonical()

        self.cyclic_path = cyclic_path
        self.lambda_max = 1
        if cyclic_path:
            self.lambda_max = 2

        self.entropy = 0

        self.n = self.mps.n
        self.d = self.mps.d
        self.omega0 = 1
        self.scale_gap = scale_gap
        self.auto_grad = auto_grad

        self.nitime = nitime

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

        self.dmax = np.max([A.shape[0] for A in tensors])

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
        self.lamb = np.clip(self.lamb + self.slope, 0, self.lambda_max)

    def get_couplings(self, lamb=None):
        if lamb is None:
            lamb = self.lamb

        wlamb = lamb
        if self.cyclic_path and wlamb > 1:
            wlamb = 2 - wlamb

        a = np.max([1 - wlamb, 0.])
        b = np.min([wlamb, 1.])
        return -a, b

    def energy_right_canonical(self, lamb=None):
        Hl0 = jnp.array([[[1.]]])
        Hl1 = jnp.array([[[1.]]])
        Hr0 = self.Hright0[0]
        Hr1 = self.Hright1[0]
        H0 = self.mpo0[0]
        H1 = self.mpo1[0]

        A = self.mps.get_tensor(0)
        a, b = self.get_couplings(lamb)
        return annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A)

    def energy_left_canonical(self, lamb=None):
        Hr0 = jnp.array([[[1.]]])
        Hr1 = jnp.array([[[1.]]])
        Hl0 = self.Hleft0[-1]
        Hl1 = self.Hleft1[-1]
        H0 = self.mpo0[-1]
        H1 = self.mpo1[-1]

        n = self.n
        A = self.mps.get_tensor(n-1)
        a, b = self.get_couplings(lamb)
        return annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A)

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

    def evolve_with_local_H(self, A, H, Hdiff, dt, omega0, omega_scale):
        if np.imag(dt) >= 1.0:
            return A, omega0, omega_scale

        val, vec = jnp.linalg.eigh(H)
        if len(val) <= 1:
            return A, omega0, omega_scale

        # calculate gaps
        # n = len(val)
        # n = 2
        # for i in range(n-1):
        #     j = i+1
        #     v1 = vec[:, i]
        #     v2 = vec[:, j]
        #     e1 = val[i]
        #     e2 = val[j]
        #     overlap = jnp.abs(jnp.conj(v1)@Hdiff@v2/(abs(e2-e1)+1e-12))
        #     omega_scale = np.max([omega_scale, overlap])

        gap = abs(val[1]-val[0])
        omega0 = np.min([omega0, gap])
        omega_scale = omega0

        if self.scale_gap:
            val = (val-val[0])/(gap + 1e-10)

        if np.imag(dt) <= -10.0:
            A = vec[:, 0]
        else:
            A = jnp.einsum("ji,j->i", jnp.conj(vec), A)
            A = jnp.einsum("i,i->i", jnp.exp(-1j*val*dt), A)
            A = jnp.einsum("ij,j->i", vec, A)

        # Adding also the derivative of the eigenstate...
        # omega_scale = omega_scale/abs(jnp.einsum("i,i->",jnp.conj(A0),vec[:,0]))

        return A, omega0, omega_scale

    def right_sweep(self, dt, lamb=None):
        Hleft0 = [jnp.array([[[1.]]])]
        Hleft1 = [jnp.array([[[1.]]])]
        # Assumes that the Hright is already prepared and that the state is in the right canonical form

        a, b = self.get_couplings(lamb)
        omega0 = jnp.inf
        omega_scale = 0

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
            Hdiff = Ha0 - Ha1
            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, Hdiff, dt, omega0, omega_scale)

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
            Hdiff = Hc0 - Hc1

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, Hdiff, -dt, omega0, omega_scale)

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
        Hdiff = Ha0 - Ha1
        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, Hdiff, dt, omega0, omega_scale)

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
        omega_scale = 0

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
            Hdiff = Ha0 - Ha1
            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, Hdiff, dt, omega0, omega_scale)

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
            Hdiff = Hc0 - Hc1

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, Hdiff, -dt, omega0, omega_scale)

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
        Hdiff = Ha0 - Ha1
        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, Hdiff, dt, omega0, omega_scale)

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
        return np.min([omega0l, omega0r]), np.max([omega_scalel, omega_scaler, 1.0])

    def apply_gradients(self, gradients, lr=1e-3):
        n = self.n
        for i in range(n):
            A = self.mps.get_tensor(i)
            A = A - lr*gradients[i]
            self.mps.set_tensor(i, A)
        self.mps.normalize()

    def single_step(self, dt, lamb, energy, energy_gradient):
        if self.auto_grad:
            ec = energy(self.mps.tensors)
            k = 0
            # self.mps.right_canonical()
            # self.Hright0 = right_context(self.mps, self.mpo0)
            # self.Hright1 = right_context(self.mps, self.mpo1)
            # omega0, omega_scale = self.right_left_sweep(dt, lamb)
            for _ in range(1000):
                gradients = energy_gradient(self.mps.tensors)
                self.apply_gradients(gradients, lr=5e-2)
                mg = np.max([jnp.linalg.norm(g) for g in gradients])
                omega0 = 1.
                omega_scale = 1.
                ec_prev = ec
                ec = energy(self.mps.tensors)
                k += 1
                # print(k,mg,abs(ec-ec_prev))
                if mg < 1e-3:
                    break
        else:
            omega0, omega_scale = self.right_left_sweep(dt, lamb)
            ec = self.energy_right_canonical(lamb)
            if abs(np.imag(dt)) >= 0:
                for istep in range(self.nitime-1):
                    ec_prev = ec
                    omega0, omega_scale = self.right_left_sweep(dt, lamb)
                    ec = self.energy_right_canonical(lamb)
                    if abs(ec-ec_prev) < 1e-6:
                        # print(istep, abs(ec-ec_prev))
                        break
        omega_scale = max([abs(omega_scale), 1e-8])
        return omega0, omega_scale, ec

    def evolve(self, data=None):
        keys = ["energy", "omega0", "omega_scale", "entropy",
                "slope", "state", "var_gs", "s", "ds_overlap", "init_overlap", "gap", "lgap", "min_gap"]
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

        if self.lamb == 0:
            k = 1
        else:
            k = int(np.ceil(self.lamb/self.ds))

        mps_prev = self.mps.copy()
        mps0 = self.mps.copy()

        if self.auto_grad:
            @jit
            def energy0(mps):
                e0 = self.energy_mpo(self.mpo0, mps)
                return e0

            @jit
            def energy1(mps):
                e1 = self.energy_mpo(self.mpo1, mps)
                return e1

            def energy(mps):
                e0 = energy0(mps)
                e1 = energy1(mps)
                a, b = self.get_couplings()
                e = a*e0+b*e1
                return e

            energy_gradient0 = jit(grad(energy0))
            energy_gradient1 = jit(grad(energy1))

            def energy_gradient(mps):
                a, b = self.get_couplings()
                gradients0 = energy_gradient0(mps)
                gradients1 = energy_gradient1(mps)
                return [a*g0+b*g1 for g0, g1 in zip(gradients0, gradients1)]
        else:
            energy = None
            energy_gradient = None

        # print("=================================")

        # lspec = linearised_specter(
        #     self.mps.tensors, self.mpo0, self.mpo1, self.Hright0, self.Hright1, lamb=0)

        # ec=self.energy_right_canonical(lamb=0)

        # print(ec,lspec[0])

        # print("=================================")

        pbar = tqdm(total=self.lambda_max, position=0, leave=True)
        pbar.update(self.lamb)
        while (self.lamb < self.lambda_max):
            self.update_lambda()
            dt = self.get_dt()
            # full step update right
            lamb = np.clip(self.lamb + self.slope, 0, self.lambda_max)
            omega0, omega_scale, ec = self.single_step(
                dt, lamb, energy, energy_gradient)
            self.omega0 = omega0

            if (self.dmax <= 0):
                lspec = linearised_specter(
                    self.mps.tensors, self.mpo0, self.mpo1, self.Hright0, self.Hright1, lamb)
                gap = np.real(lspec[0][0]-ec)
                data["gap"].append(gap)
                spec = np.real(lspec[0])
                gaps = np.diff(spec)
                lgap = gaps[0]
                min_gap = np.min(gaps)
                data["lgap"].append(lgap)
                data["min_gap"].append(min_gap)
                # omega_scale = np.abs(lgap)

                # print("omega", omega0, gap, lgap, min_gap,  omega0 - np.real(lspec[0][0]-ec))

            # print("init overlap",1-abs(self.mps.overlap(mps0)))
            # print(lamb,np.real(ec),lspec[0])
            # print(ec, energy(self.mps.tensors))
            # print("gradient", [jnp.linalg.norm(g) for g in energy_gradient(self.mps.tensors)])

            if self.adaptive and not self.auto_grad:
                self.slope = np.clip(
                    self.slope_omega*omega_scale, self.min_slope, self.max_slope)

            data["omega0"].append(float(np.real(omega0)))
            data["omega_scale"].append(float(np.real(omega_scale)))

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
            # self.update_lambda()

            tcurrent = time.time()
            if tcurrent-self.tstart > 3600*47 or self.killer.kill_now:
                print(
                    f"Killing program after {int(tcurrent-self.tstart)} seconds.")
                break
        data["mps"] = [np.array(A) for A in self.mps.tensors]
        pbar.close()
        return data
