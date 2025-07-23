from jax import jit, grad
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import svd
from tqdm import tqdm
from GracefulKiller import GracefulKiller
import time
import numpy as np


from tdvp_qa.mps import MPS
from tdvp_qa.utils import annealing_energy_canonical, right_hamiltonian, left_hamiltonian, right_context, effective_hamiltonian_A, effective_hamiltonian_C, linearised_specter, right_context_mps
from tdvp_qa.utils import effective_hamiltonian_A_MPS, left_hamiltonian_mps, right_hamiltonian_mps, effective_hamiltonian_C_MPS


@jit
def _energy_mpo(A, O, e, nrm):
    e = jnp.einsum("umd,uiU->Umid", e, A)
    e = jnp.einsum("Umid,djD->UmijD", e, jnp.conj(A))
    e = jnp.einsum("UmijD,mijM->UMD", e, O)

    nrm = jnp.einsum("ud,uiU->Uid", nrm, A)
    nrm = jnp.einsum("Uid,diD->UD", nrm, jnp.conj(A))
    return e, nrm

@jit
def local_prods(A, o0, o1, o2):
    op02 = jnp.einsum("lijr,LjJR->lLiJrR", o0, o2)
    op20 = jnp.einsum("lijr,LjJR->lLiJrR", o2, o0)
    op12 = jnp.einsum("lijr,LjJR->lLiJrR", o1, o2)
    op21 = jnp.einsum("lijr,LjJR->lLiJrR", o2, o1)
    op02s = jnp.einsum("lLiJrR,aJb->lLairRb", op02, A)
    op20s = jnp.einsum("lLiJrR,aJb->lLairRb", op20, A)
    op12s = jnp.einsum("lLiJrR,aJb->lLairRb", op12, A)
    op21s = jnp.einsum("lLiJrR,aJb->lLairRb", op21, A)
    op0s = jnp.einsum("lijr,ajb->lairb", o0, A)
    op1s = jnp.einsum("lijr,ajb->lairb", o1, A)
    return op02s, op20s, op12s, op21s, op0s, op1s


@jit
def multiply_ops_right(op11, op12, op13, op14, op15, op16,
                    op22, op23, op24, op25, op26,
                    op33, op34, op35, op36,
                    op44, op45, op46,
                    op55, op56, op66,
                    op1s, op2s, op3s, op4s, op5s, op6s):
    # 11
    op11 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op11, jnp.conj(op1s))
    op11 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op11, op1s)
    # 12
    op12 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op12, jnp.conj(op1s))
    op12 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op12, op2s)
    # 13
    op13 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op13, jnp.conj(op1s))
    op13 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op13, op3s)
    # 14
    op14 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op14, jnp.conj(op1s))
    op14 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op14, op4s)

    # 22
    op22 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op22, jnp.conj(op2s))
    op22 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op22, op2s)
    # 23
    op23 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op23, jnp.conj(op2s))
    op23 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op23, op3s)
    # 24
    op24 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op24, jnp.conj(op2s))
    op24 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op24, op4s)

    # 33
    op33 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op33, jnp.conj(op3s))
    op33 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op33, op3s)
    # 34
    op34 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op34, jnp.conj(op3s))
    op34 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op34, op4s)

    # 44
    op44 = jnp.einsum("lLarRb,lLaitTc->tTcrRbi", op44, jnp.conj(op4s))
    op44 = jnp.einsum("tTcrRbi,rRbivVd->tTcvVd", op44, op4s)

    # 15
    op15 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op15, jnp.conj(op1s))
    op15 = jnp.einsum("tTcrbi,rbivd->tTcvd", op15, op5s)

    # 16
    op16 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op16, jnp.conj(op1s))
    op16 = jnp.einsum("tTcrbi,rbivd->tTcvd", op16, op6s)

    # 25
    op25 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op25, jnp.conj(op2s))
    op25 = jnp.einsum("tTcrbi,rbivd->tTcvd", op25, op5s)

    # 26
    op26 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op26, jnp.conj(op2s))
    op26 = jnp.einsum("tTcrbi,rbivd->tTcvd", op26, op6s)

    # 35
    op35 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op35, jnp.conj(op3s))
    op35 = jnp.einsum("tTcrbi,rbivd->tTcvd", op35, op5s)

    # 36
    op36 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op36, jnp.conj(op3s))
    op36 = jnp.einsum("tTcrbi,rbivd->tTcvd", op36, op6s)

    # 45
    op45 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op45, jnp.conj(op4s))
    op45 = jnp.einsum("tTcrbi,rbivd->tTcvd", op45, op5s)

    # 46
    op46 = jnp.einsum("lLarb,lLaitTc->tTcrbi", op46, jnp.conj(op4s))
    op46 = jnp.einsum("tTcrbi,rbivd->tTcvd", op46, op6s)

    # if calcAll:
    # 55
    op55 = jnp.einsum("larb,laitc->tcrbi", op55, jnp.conj(op5s))
    op55 = jnp.einsum("tcrbi,rbivd->tcvd", op55, op5s)
    # 56
    op56 = jnp.einsum("larb,laitc->tcrbi", op56, jnp.conj(op5s))
    op56 = jnp.einsum("tcrbi,rbivd->tcvd", op56, op6s)
    # 66
    op66 = jnp.einsum("larb,laitc->tcrbi", op66, jnp.conj(op6s))
    op66 = jnp.einsum("tcrbi,rbivd->tcvd", op66, op6s)

    return op11, op12, op13, op14, op15, op16, op22, op23, op24, op25, op26, op33, op34, op35, op36, op44, op45, op46, op55, op56, op66

class TDVP_AGP():
    def __init__(self, mpo0, mpo1, mpo2, tensorslist, slope, dt, lamb=0, max_slope=0.05, min_slope=1e-6, adaptive=False, compute_states=False, key=42, slope_omega=1e-3, ds=0.01, scale_gap=False, nitime=10, auto_grad=False, cyclic_path=False, Tmc=None, nmps=1, reorder_mps=False):
        # mpo0, mpo1 are simple nxMxdxdxM tensors containing the MPO representations of H0 and H1
        self.mpo0 = [jnp.array(A) for A in mpo0]
        self.mpo1 = [jnp.array(A) for A in mpo1]
        # mpo2 is a simple nxMxdxdxM tensors containing the adiabatic gauge potential (has to be computed on the fly)
        self.mpo2 = [jnp.array(A) for A in mpo2]

        # The MPS at initialization should be in the right canonical form
        self.nmps = nmps
        self.reorder_mps = reorder_mps

        self.mpslist = []
        for tensors in tensorslist:
            mps = MPS(tensors, key)
            mps.right_canonical()
            self.mpslist.append(mps)

        self.T = Tmc

        self.cyclic_path = cyclic_path
        self.lambda_max = 1
        if cyclic_path:
            self.lambda_max = 2

        self.entropy = [0 for _ in range(nmps)]

        self.n = self.mpslist[0].n
        self.d = self.mpslist[0].d
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

        self.Hright0 = [right_context(mps, self.mpo0) for mps in self.mpslist]
        self.Hright1 = [right_context(mps, self.mpo1) for mps in self.mpslist]
        self.Hright2 = [right_context(mps, self.mpo2) for mps in self.mpslist]

        self.Hleft0 = [None for _ in range(self.nmps)]
        self.Hleft1 = [None for _ in range(self.nmps)]
        self.Hleft2 = [None for _ in range(self.nmps)]

        self.HrightMPS = []
        for imps in range(self.nmps):
            hr = [right_context_mps(self.mpslist[imps], self.mpslist[j])
                  for j in range(imps)]
            self.HrightMPS.append(hr)
        self.HleftMPS = [None for _ in range(self.nmps)]

        self.dmax = np.max([A.shape[0] for A in tensors])

        self.key = random.PRNGKey(key)

        self.tstart = time.time()
        self.killer = GracefulKiller()

    def get_dt(self):
        # key, subkey = random.split(self.key)
        # dtr = jnp.real(self.dt)
        # dti = jnp.imag(self.dt) * random.uniform(subkey)
        # dt = dtr + 1j*dti
        # self.key = key
        return self.dt

    def update_lambda(self):
        self.lamb = np.clip(self.lamb + self.slope, 0, self.lambda_max)

    def get_couplings(self, s=None):
        if s is None:
            s = self.s

        wlamb = lamb
        if self.cyclic_path and wlamb > 1:
            wlamb = 2 - wlamb

        a = np.max([1 - wlamb, 0.])
        b = np.min([wlamb, 1.])
        c = 4*np.clip((1-wlamb)*wlamb, 0., 1.)
        return -a, b, c

    def energy_right_canonical(self, lamb=None):
        Hl0 = jnp.array([[[1.]]])
        Hl1 = jnp.array([[[1.]]])
        H0 = self.mpo0[0]
        H1 = self.mpo1[0]

        elist = []
        for imps in range(self.nmps):
            Hr0 = self.Hright0[imps][0]
            Hr1 = self.Hright1[imps][0]
            A = self.mpslist[imps].get_tensor(0)
            a, b, _ = self.get_couplings(lamb)
            elist.append(annealing_energy_canonical(
                Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A))
        return elist

    def energy_left_canonical(self, lamb=None):
        Hr0 = jnp.array([[[1.]]])
        Hr1 = jnp.array([[[1.]]])
        Hl0 = self.Hleft0[-1]
        Hl1 = self.Hleft1[-1]
        H0 = self.mpo0[-1]
        H1 = self.mpo1[-1]

        elist = []
        n = self.n
        for imps in range(self.nmps):
            A = self.mpslist[imps].get_tensor(n-1)
            a, b = self.get_couplings(lamb)
            elist.append(annealing_energy_canonical(
                Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A))
        return elist

    def energy_mpo(self, mpo, mps):
        e = jnp.array([[[1.0]]])
        n = self.n
        nrm = jnp.array([[1.0]])
        for i in range(n):
            A = mps[i]
            O = mpo[i]
            e, nrm = _energy_mpo(A, O, e, nrm)

        e = jnp.real(e[0, 0, 0])
        nrm = jnp.real(nrm[0, 0])
        return e/nrm

    def evolve_with_local_H(self, A, H, dt, omega0, omega_scale):
        if np.imag(dt) >= 1.0:
            return A, omega0, omega_scale

        val, vec = jnp.linalg.eigh(H)
        if len(val) <= 1:
            return A, omega0, omega_scale

        gap = abs(val[1]-val[0])
        omega0 = np.min([omega0, gap])
        omega_scale = omega0

        if self.scale_gap:
            val = (val-val[0])/(gap + 1e-10)

        if self.T and self.T > 0:
            key, subkey = random.split(self.key)
            val = val - jnp.min(val)
            probs = jnp.exp(-val/self.T)
            cprob = jnp.cumsum(probs)
            r = random.uniform(subkey) * cprob[-1]
            self.key = key

            for i, cp in enumerate(cprob):
                if r < cp:
                    A = vec[:, i]
                    break
        elif np.imag(dt) <= -10.0 or self.T and np.isclose(self.T, 0.):
            A = vec[:, 0]
        else:
            max_abs = 100
            val = val - jnp.min(val)
            max_val = jnp.max(jnp.abs(val))
            if max_val > max_abs:
                val = max_abs*val/max_val

            A = jnp.einsum("ji,j->i", jnp.conj(vec), A)
            A = jnp.einsum("i,i->i", jnp.exp(-1j*val*dt), A)
            A = jnp.einsum("ij,j->i", vec, A)

        # Adding also the derivative of the eigenstate...
        # omega_scale = omega_scale/abs(jnp.einsum("i,i->",jnp.conj(A0),vec[:,0]))

        return A, omega0, omega_scale

    def right_sweep(self, dt, lamb=None):
        for imps in np.arange(self.nmps):
            omega0, omega_scale = self.right_sweep_imps(
                dt, lamb=lamb, imps=imps)
        return omega0, omega_scale

    def right_sweep_imps(self, dt, lamb=None, imps=0):
        Hleft0 = [jnp.array([[[1.]]])]
        Hleft1 = [jnp.array([[[1.]]])]

        HleftMPS = [[jnp.array([[1.0]])] for _ in range(imps)]
        # Assumes that the Hright is already prepared and that the state is in the right canonical form

        a, b, c = self.get_couplings(lamb)
        omega0 = jnp.inf
        omega_scale = 0

        n = self.n

        for i in range(n-1):
            Hl0 = Hleft0[i]
            Hl1 = Hleft1[i]
            Hr0 = self.Hright0[imps][i]
            Hr1 = self.Hright1[imps][i]
            H0 = self.mpo0[i]
            H1 = self.mpo1[i]

            A = self.mpslist[imps].get_tensor(i)
            Dl, d, Dr = A.shape

            # Effective Hamiltonian for A
            dd = Dl*d*Dr
            Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
            Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])

            Ha = a * Ha0 + b * Ha1

            # Adding the MPS-catalyst parts
            if imps > 0:
                Hmps = 0
                for j in range(imps):
                    hl = HleftMPS[j][i]
                    hr = self.HrightMPS[imps][j][i]
                    B = self.mpslist[j].get_tensor(i)
                    Hmps += jnp.reshape(effective_hamiltonian_A_MPS(hl,
                                        B, hr), [dd, dd])
                Ha += c * Hmps

            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, dt, omega0, omega_scale)

            A = jnp.reshape(A, [Dl, d, Dr])

            A = A/jnp.linalg.norm(A)
            self.mpslist[imps].set_tensor(i, A)
            r, Ar = self.mpslist[imps].move_right(i)

            # Calculating new Hleft
            A = self.mpslist[imps].get_tensor(i)
            Hl0 = left_hamiltonian(A, Hleft0[i], H0)
            Hl1 = left_hamiltonian(A, Hleft1[i], H1)
            Hleft0.append(Hl0)
            Hleft1.append(Hl1)

            # New HleftMPS
            if imps > 0:
                for j in range(imps):
                    B = self.mpslist[j].get_tensor(i)
                    Hlmps = HleftMPS[j][i]
                    Hlmps = left_hamiltonian_mps(Hlmps, B, A)
                    HleftMPS[j].append(Hlmps)

            # Effective Hamiltonian for C
            Dl, Dr = r.shape
            Hc0 = jnp.reshape(effective_hamiltonian_C(
                Hl0, Hr0), [Dl*Dr, Dl*Dr])
            Hc1 = jnp.reshape(effective_hamiltonian_C(
                Hl1, Hr1), [Dl*Dr, Dl*Dr])

            Hc = a * Hc0 + b * Hc1

            # Adding the MPS-catalyst part
            if imps > 0:
                Hmps = 0
                for j in range(imps):
                    hl = HleftMPS[j][-1]
                    hr = self.HrightMPS[imps][j][i]
                    Hmps += jnp.reshape(effective_hamiltonian_C_MPS(hl,
                                        hr), [Dl*Dr, Dl*Dr])
                Hc += c*Hmps

            C = jnp.reshape(r, [Dl*Dr])

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, -dt, omega0, omega_scale)

            C = jnp.reshape(C, [Dl, Dr])
            C = C/jnp.linalg.norm(C)
            Ar = jnp.einsum("ij,jkl->ikl", C, Ar)
            self.mpslist[imps].set_tensor(i+1, Ar)

        #  Handling the last site update

        i = n-1
        Hl0 = Hleft0[i]
        Hl1 = Hleft1[i]
        Hr0 = self.Hright0[imps][i]
        Hr1 = self.Hright1[imps][i]
        H0 = self.mpo0[i]
        H1 = self.mpo1[i]

        A = self.mpslist[imps].get_tensor(i)
        Dl, d, Dr = A.shape

        # Effective Hamiltonian for A
        dd = Dl*d*Dr
        Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        Ha = a * Ha0 + b * Ha1

        # Adding the MPS-catalyst parts
        if imps > 0:
            Hmps = 0
            for j in range(imps):
                hl = HleftMPS[j][-1]
                hr = self.HrightMPS[imps][j][i]
                B = self.mpslist[j].get_tensor(i)
                Hmps += jnp.reshape(effective_hamiltonian_A_MPS(hl,
                                    B, hr), [dd, dd])
            Ha += c * Hmps

        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, dt, omega0, omega_scale)

        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mpslist[imps].set_tensor(i, A)

        # Setting left effective Hamiltonians
        self.Hleft0[imps] = Hleft0
        self.Hleft1[imps] = Hleft1
        self.HleftMPS[imps] = HleftMPS

        return omega0, omega_scale

    def left_sweep(self, dt, lamb=None):
        for imps in np.arange(self.nmps):
            omega0, omega_scale = self.left_sweep_imps(
                dt, lamb=lamb, imps=imps)
        return omega0, omega_scale

    def left_sweep_imps(self, dt, lamb=None, imps=0):
        Hright0 = [jnp.array([[[1.]]])]
        Hright1 = [jnp.array([[[1.]]])]
        HrightMPS = [[jnp.array([[1.]])] for _ in range(imps)]
        # Assumes that the Hleft is already prepared and that the state is in the left canonical form

        n = self.n
        a, b, c = self.get_couplings(lamb)
        omega0 = jnp.inf
        omega_scale = 0

        for i in range(n-1, 0, -1):
            Hl0 = self.Hleft0[imps][i]
            Hl1 = self.Hleft1[imps][i]
            Hr0 = Hright0[0]
            Hr1 = Hright1[0]
            H0 = self.mpo0[i]
            H1 = self.mpo1[i]

            A = self.mpslist[imps].get_tensor(i)
            Dl, d, Dr = A.shape

            # Effective Hamiltonian for A
            dd = Dl*d*Dr
            Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
            Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
            Ha = a * Ha0 + b * Ha1

            # Adding the MPS-catalyst parts
            if imps > 0:
                Hmps = 0
                for j in range(imps):
                    hr = HrightMPS[j][0]
                    hl = self.HleftMPS[imps][j][i]
                    B = self.mpslist[j].get_tensor(i)
                    Hmps += jnp.reshape(effective_hamiltonian_A_MPS(hl,
                                        B, hr), [dd, dd])
                Ha += c * Hmps

            A = jnp.reshape(A, [dd])

            # Updating A
            A, omega0, omega_scale = self.evolve_with_local_H(
                A, Ha, dt, omega0, omega_scale)

            A = jnp.reshape(A, [Dl, d, Dr])

            A = A/jnp.linalg.norm(A)
            self.mpslist[imps].set_tensor(i, A)
            Al, r = self.mpslist[imps].move_left(i)

            # Calculating new Hright
            A = self.mpslist[imps].get_tensor(i)
            Hr0 = right_hamiltonian(A, Hr0, H0)
            Hr1 = right_hamiltonian(A, Hr1, H1)
            Hright0 = [Hr0] + Hright0
            Hright1 = [Hr1] + Hright1

            # New HrightMPS
            if imps > 0:
                for j in range(imps):
                    B = self.mpslist[j].get_tensor(i)
                    Hrmps = HrightMPS[j][0]
                    Hrmps = right_hamiltonian_mps(Hrmps, B, A)
                    HrightMPS[j] = [Hrmps] + HrightMPS[j]

            # Effective Hamiltonian for C
            Dl, Dr = r.shape
            Hc0 = jnp.reshape(effective_hamiltonian_C(
                Hl0, Hr0), [Dl*Dr, Dl*Dr])
            Hc1 = jnp.reshape(effective_hamiltonian_C(
                Hl1, Hr1), [Dl*Dr, Dl*Dr])
            Hc = a * Hc0 + b * Hc1

            # Adding the MPS-catalyst part
            if imps > 0:
                Hmps = 0
                for j in range(imps):
                    hr = HrightMPS[j][0]
                    hl = self.HleftMPS[imps][j][i]
                    Hmps += jnp.reshape(effective_hamiltonian_C_MPS(hl,
                                        hr), [Dl*Dr, Dl*Dr])
                Hc += c*Hmps

            C = jnp.reshape(r, [Dl*Dr])

            # Updating C
            C, omega0, omega_scale = self.evolve_with_local_H(
                C, Hc, -dt, omega0, omega_scale)

            C = jnp.reshape(C, [Dl, Dr])
            C = C/jnp.linalg.norm(C)
            Al = jnp.einsum("ijk,kl->ijl", Al, C)

            # Calculating the entropy on the fly
            if i == n//2:
                _, s, _ = svd(C)
                self.entropy[imps] = -np.log(s) @ s

            self.mpslist[imps].set_tensor(i-1, Al)

        # Updating the first site
        i = 0
        Hl0 = self.Hleft0[imps][i]
        Hl1 = self.Hleft0[imps][i]
        Hr0 = Hright0[0]
        Hr1 = Hright1[0]
        H0 = self.mpo0[i]
        H1 = self.mpo1[i]

        A = self.mpslist[imps].get_tensor(i)
        Dl, d, Dr = A.shape

        # Effective Hamiltonian for A
        dd = Dl*d*Dr
        Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
        Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
        Ha = a * Ha0 + b * Ha1

        # Adding the MPS-catalyst parts
        if imps > 0:
            Hmps = 0
            for j in range(imps):
                hr = HrightMPS[j][0]
                hl = self.HleftMPS[imps][j][i]
                B = self.mpslist[j].get_tensor(i)
                Hmps += jnp.reshape(effective_hamiltonian_A_MPS(hl,
                                    B, hr), [dd, dd])
            Ha += c * Hmps

        A = jnp.reshape(A, [dd])

        # Updating A
        A, omega0, omega_scale = self.evolve_with_local_H(
            A, Ha, dt, omega0, omega_scale)

        A = jnp.reshape(A, [Dl, d, Dr])
        A = A/jnp.linalg.norm(A)
        self.mpslist[imps].set_tensor(i, A)

        # Setting right effective Hamiltonians
        self.Hright0[imps] = Hright0
        self.Hright1[imps] = Hright1
        self.HrightMPS[imps] = HrightMPS

        return omega0, omega_scale

    def right_left_sweep(self, dt, lamb=None):
        omega0r, omega_scaler = self.right_sweep(dt/2., lamb)
        omega0l, omega_scalel = self.left_sweep(dt/2., lamb)
        return np.min([omega0l, omega0r]), np.max([omega_scalel, omega_scaler, 1.0])

    def apply_gradients(self, gradients, lr=1e-3):
        raise NotImplementedError
        # n = self.n
        # for i in range(n):
        #     A = self.mps.get_tensor(i)
        #     A = A - lr*gradients[i]
        #     self.mps.set_tensor(i, A)
        # self.mps.normalize()

    def single_step(self, dt, lamb, energy, energy_gradient):
        if self.auto_grad:
            raise NotImplementedError
        else:
            omega0, omega_scale = self.right_left_sweep(dt, lamb)
            eclist = self.energy_right_canonical(lamb)
            if abs(np.imag(dt)) > 0 or self.T and np.isclose(self.T, 0.):
                for _ in range(self.nitime-1):
                    ec_prev = np.min(eclist)
                    omega0, omega_scale = self.right_left_sweep(dt, lamb)
                    eclist = self.energy_right_canonical(lamb)
                    ec = np.min(eclist)
                    if abs(ec-ec_prev) < 1e-6:
                        # print(istep, abs(ec-ec_prev))
                        break

        eclist = self.reorder_mpslist(eclist)
        omega_scale = max([abs(omega_scale), 1e-8])
        return omega0, omega_scale, eclist

    def reorder_mpslist(self, eclist):
        elist = [float(np.real(ec)) for ec in eclist]
        if not self.reorder_mps:
            return elist
        nmps = self.nmps
        mpslist = [None for _ in range(nmps)]
        entropy = [None for _ in range(nmps)]
        eclist = [None for _ in range(nmps)]
        inds = np.argsort(elist)
        for i in range(nmps):
            mpslist[i] = self.mpslist[inds[i]]
            entropy[i] = self.entropy[inds[i]]
            eclist[i] = elist[inds[i]]
        self.mpslist = mpslist
        self.entropy = entropy
        return eclist
    
    def agpf(self,mpo2):
        mps = self.mps 
        mpo0 = self.mpo0
        mpo1 = self.mpo1
        a,b,c = self.get_couplings()
        n = self.n
        op11 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op12 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op13 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op14 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op22 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op23 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op24 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op33 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op34 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))
        op44 = jnp.array(np.ones([1, 1, 1, 1, 1, 1]))

        op15 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op16 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op25 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op26 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op35 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op36 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op45 = jnp.array(np.ones([1, 1, 1, 1, 1]))
        op46 = jnp.array(np.ones([1, 1, 1, 1, 1]))

        op55 = jnp.array(np.ones([1, 1, 1, 1]))
        op56 = jnp.array(np.ones([1, 1, 1, 1]))
        op66 = jnp.array(np.ones([1, 1, 1, 1]))

        for i in range(n):
            op1s, op2s, op3s, op4s, op5s, op6s = local_prods(
                mps[i], mpo0[i], mpo1[i], mpo2[i])
            op11, op12, op13, op14, op15, op16, op22, op23, op24, op25, op26, op33, op34, op35, op36, op44, op45, op46, op55, op56, op66 = multiply_ops_right(op11, op12, op13, op14, op15, op16,
                                                                                                                                                            op22, op23, op24, op25, op26,
                                                                                                                                                            op33, op34, op35, op36,
                                                                                                                                                            op44, op45, op46,
                                                                                                                                                            op55, op56, op66,
                                                                                                                                                            op1s, op2s, op3s, op4s, op5s, op6s)
        op11 = jnp.real(op11[0, 0, 0, 0, 0, 0])
        op12 = 2*jnp.real(op12[0, 0, 0, 0, 0, 0])
        op13 = 2*jnp.real(op13[0, 0, 0, 0, 0, 0])
        op14 = 2*jnp.real(op14[0, 0, 0, 0, 0, 0])
        op15 = 2*jnp.real(op15[0, 0, 0, 0, 0])
        op16 = 2*jnp.real(op16[0, 0, 0, 0, 0])

        op22 = jnp.real(op22[0, 0, 0, 0, 0, 0])
        op23 = 2*jnp.real(op23[0, 0, 0, 0, 0, 0])
        op24 = 2*jnp.real(op24[0, 0, 0, 0, 0, 0])
        op25 = 2*jnp.real(op25[0, 0, 0, 0, 0])
        op26 = 2*jnp.real(op26[0, 0, 0, 0, 0])

        op33 = jnp.real(op33[0, 0, 0, 0, 0, 0])
        op34 = 2*jnp.real(op34[0, 0, 0, 0, 0, 0])
        op35 = 2*jnp.real(op35[0, 0, 0, 0, 0])
        op36 = 2*jnp.real(op36[0, 0, 0, 0, 0])

        op44 = jnp.real(op44[0, 0, 0, 0, 0, 0])
        op45 = 2*jnp.real(op45[0, 0, 0, 0, 0])
        op46 = 2*jnp.real(op46[0, 0, 0, 0, 0])

        op55 = jnp.real(op55[0, 0, 0, 0])
        op56 = 2*jnp.real(op56[0, 0, 0, 0])

        op66 = jnp.real(op66[0, 0, 0, 0])

        agpl = op11 + op12 + op13 + op14 + op15 + op16 + \
            op22 + op23 + op24 + op25 + op26 +  \
            op33 + op34 + op35 + op36 + \
            op44 + op45 + op46 + \
            op55 + op56 + op66

        return agpl, op11, op12, op13, op14, op15, op16, op22, op23, op24, op25, op26, op33, op34, op35, op36, op44, op45, op46, op55, op56, op66

    def evolve(self, data=None):
        keys = ["energy", "omega0", "omega_scale", "entropy",
                "slope", "state", "var_gs", "s", "ds_overlap", "init_overlap", "gap", "lgap", "min_gap", "var_e"]
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

        if self.s == 0:
            k = 1
        else:
            k = int(np.ceil(self.s/self.ds))

        energy = None
        energy_gradient = None


        @jit
        def agp_loss(mpo2):
            agpl, op11, op12, op13, op14, op15, op16, op22, op23, op24, op25, op26, op33, op34, op35, op36, op44, op45, op46, op55, op56, op66 = self.agpf(mpo2)
            return agpl

        agp_grad = jit(grad(agp_loss))

        def apply_grads(lamb=1e-12):
            gradients = agp_grad(self.mpo2)
            n = self.n
            for i in range(n):
                g = gradients[i]
                self.mpo2[i] = self.mpo2[i] - lamb * g
    
        def update_mpo2(nstep=1000):
            agpls = []
            a = agp_loss(self.mpo2)
            lamb =1e-3/abs(a) 
            for i in range(nstep):
                a = agp_loss(self.mpo2)
                # a =  agp_loss_all(mpo2)
                agpls.append(a)
                if a[0]<1e-5:
                    break
                apply_grads(lamb=np.min([1e-3/np.abs(a),1e-4]))
            return a

        a=update_mpo2(1000)
        print(a)

        pbar = tqdm(total=self.lambda_max, position=0, leave=True)
        pbar.update(self.lamb)
        while (self.lamb < self.lambda_max):
            self.update_lambda()
            dt = self.get_dt()
            # full step update right
            lamb = np.clip(self.lamb + self.slope, 0, self.lambda_max)
            omega0, omega_scale, eclist = self.single_step(
                dt, lamb, energy, energy_gradient)
            self.omega0 = omega0

            if self.adaptive and not self.auto_grad:
                self.slope = np.clip(
                    self.slope_omega*omega_scale, self.min_slope, self.max_slope)

            data["omega0"].append(float(np.real(omega0)))
            data["omega_scale"].append(float(np.real(omega_scale)))

            if lamb >= k*self.ds:
                data["energy"].append(eclist)
                data["entropy"].append([
                    float(np.real(ent/np.log(2.0))) for ent in self.entropy])
                data["s"].append(lamb)
                # data["ds_overlap"].append(abs(self.mps.overlap(mps_prev)))
                # data["init_overlap"].append(abs(self.mps.overlap(mps0)))
                # mps_prev = self.mps.copy()
                k = k+1
                if self.compute_states:
                    vargs = []
                    ves = []
                    for i, mps in enumerate(self.mpslist):
                        dmrg_mps = mps.copy()
                        dmrg_mps.dmrg(1, self.mpo0, self.mpo1,
                                      self.Hright0[i], self.Hright1[i], sweeps=20)
                        vargs.append([np.array(A)
                                      for A in dmrg_mps.tensors])
                        ve = self.energy_mpo(self.mpo1, dmrg_mps.tensors)
                        ves.append(ve)

                    data["var_gs"].append(vargs)
                    data["var_e"].append(ves)
                    data["state"].append(
                        [self.mpslist[imps].tensors for imps in range(self.nmps)])
                    # raise NotImplementedError
            data["slope"].append(float(np.real(self.slope)))
            pbar.update(float(np.real(self.slope)))
            # self.update_lambda()

            tcurrent = time.time()
            if tcurrent-self.tstart > 3600*47 or self.killer.kill_now:
                print(
                    f"Killing program after {int(tcurrent-self.tstart)} seconds.")
                break
        pbar.close()

        # Adding the variational ground state obtained with DMRG when starting from the final state
        data["last_var_gs"] = []
        for imps in range(self.nmps):
            dmrg_mps = self.mpslist[imps].copy()
            dmrg_mps.dmrg(lamb, self.mpo0, self.mpo1,
                          self.Hright0[imps], self.Hright1[imps], sweeps=20)
            # data["var_gs"].append(np.array(dmrg_mps.construct_state()))
            data["last_var_gs"].append([np.array(A)
                                        for A in dmrg_mps.tensors])
        return data
