from jax import jit
import jax.numpy as jnp
from numpy import max, min, prod


# @jit
def construct_state(mps):
    n = len(mps)
    assert n <= 12, f"Can not construct a state with more than 12 spins. The state has {n} spins"
    psi = mps[0]
    print(len(mps))
    for i in range(1, n):
        psi = jnp.einsum("...i,ijk", psi, mps[i])
    return jnp.reshape(psi, [-1])


def construct_operator(mpo):
    n = len(mpo)
    assert n <= 12, f"Can not construct a state with more than 12 spins. The state has {n} spins"
    psi = mpo[0]
    for i in range(1, n):
        psi = jnp.einsum("...i,ijkl->...jkl", psi, mpo[i])
    psi = jnp.squeeze(psi)
    dims = psi.shape
    ord = [2*i for i in range(n)] + [2*i+1 for i in range(n)]
    psi = jnp.transpose(psi, ord)
    return jnp.reshape(psi, [-1, prod(dims[::2])])


def annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, a, b, A):
    Dl, d, Dr = A.shape
    # a = -max([1 - lamb, 0.])
    # b = min([lamb, 1.])

    # Effective Hamiltonian for A
    dd = Dl*d*Dr
    Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
    Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])
    Ha = a * Ha0 + b * Ha1

    A = jnp.reshape(A, [-1])
    return jnp.einsum("i,ij,j", jnp.conj(A), Ha, A)


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


@jit
def left_hamiltonian_mps(Hlmps, B, A):
    Hlmps = jnp.einsum("Ll,LsR->Rls", Hlmps, jnp.conj(B))
    Hlmps = jnp.einsum("Rls,lsr->Rr", Hlmps, A)
    return Hlmps


@jit
def right_hamiltonian_mps(Hrmps, B, A):
    Hrmps = jnp.einsum("LsR,Rr->Lsr", jnp.conj(B), Hrmps)
    Hrmps = jnp.einsum("Lsr,lsr->Ll", Hrmps, A)
    return Hrmps


def right_context_c(mpsc, mpo):
    # we assume that the MPS is in the central canonical form:
    raise NotImplementedError


def left_context_c(mpsc, mpo):
    # we assume that the MPS is in the central canonical form:
    raise NotImplementedError


def right_context(mps, mpo):
    # Here we assume that the mps is already in the right canonical form
    n = mps.n
    Hright = [jnp.array([[[1.]]])]
    for i in range(n-1, 0, -1):
        H0 = mpo[i]
        A = mps.get_tensor(i)
        Hr = right_hamiltonian(A, Hright[0], H0)
        Hright = [Hr] + Hright
    return Hright


def right_context_mps(mps, Hmps):
    # Here we assume that the mps is already in the right canonical form
    n = mps.n
    Hright = [jnp.array([[1.]])]
    for i in range(n-1, 0, -1):
        H0 = Hright[0]
        A = mps.get_tensor(i)
        B = Hmps.get_tensor(i)
        Hr = jnp.einsum("LsR,lsr,Rr->Ll", jnp.conj(B), A, H0)
        Hright = [Hr] + Hright
    return Hright


@jit
def effective_hamiltonian_A_MPS(hl, A, hr):
    B = jnp.einsum("Ll,LsR->lsR", hl, A)
    B = jnp.einsum("lsR,Rr->lsr", B, hr)
    B = jnp.einsum("LSR,lsr->LSRlsr", jnp.conj(B), B)
    return B


@jit
def effective_hamiltonian_A(Hl, Hr, H0):
    Heff = jnp.einsum("umd,mijn->diujn", Hl, H0)
    Heff = jnp.einsum("diujn,anb->dibuja", Heff, Hr)
    return Heff


@jit
def effective_hamiltonian_C_MPS(hl, hr):
    H = jnp.einsum("Ll,Rr->lr", hl, hr)
    H = jnp.einsum("LR,lr->LRlr", jnp.conj(H), H)
    return H


@jit
def effective_hamiltonian_C(Hl, Hr):
    Heff = jnp.einsum("umd,amb->dbua", Hl, Hr)
    return Heff


def full_effective_hamiltonian_A(Hl0, Hl1, H0, H1, Hr0, Hr1, lamb, dd):
    a = -max([1 - lamb, 0.])
    b = min([lamb, 1.])

    # Effective Hamiltonian for A
    Ha0 = jnp.reshape(effective_hamiltonian_A(Hl0, Hr0, H0), [dd, dd])
    Ha1 = jnp.reshape(effective_hamiltonian_A(Hl1, Hr1, H1), [dd, dd])

    Ha0 = jnp.reshape(Ha0, [dd, dd])
    Ha1 = jnp.reshape(Ha1, [dd, dd])

    Ha = a * Ha0 + b * Ha1
    return Ha


def linearised_hamiltonian(mps, lv, tv, mpo, Hright):
    # We assume that the mps is in the right canonical form
    n = len(mps)
    Hl = jnp.array([[[1.]]])
    dims = [prod(A.shape) for A in mps]
    Klist = []
    for i in range(n):
        # Diagonal of K
        Kl = jnp.einsum("iaj,adub->jdiub", Hl, mpo[i])
        K = jnp.einsum("jdiub,kbl->jdliuk", Kl, Hright[i])
        Kij = []
        for j in range(i):
            Kij.append(Klist[j][i])
        Kij.append(K)
        # Move Hl to the right
        A = lv[i]
        Hl = left_hamiltonian(A, Hl, mpo[i])
        # Preparing first off-diagonal Kl
        Kl = jnp.einsum("jdiub,iuk->jdkb", Kl, A)
        if i < n-1:
            Kl = jnp.einsum("JDia,adub->JDiudb", Kl, mpo[i+1])
            Kl = jnp.einsum("JDiudb,Ldl->JDLiubl", Kl, jnp.conj(mps[i+1]))

        # Calculate offdiagonal elements of K
        for j in range(i+1, n):
            # Calculating the offdiagonal K
            K = jnp.einsum("JDLiubl,kbl->JDLiuk", Kl, Hright[j])
            Kij.append(K)

            # Calculating the new Kl and moving A2 to right
            A = lv[j]
            if j < n-1:
                Kl = jnp.einsum("JDLkual,kui->JDLial", Kl, A)
                Kl = jnp.einsum("JDLiar,rdl->JDLliad", Kl, jnp.conj(mps[j+1]))
                Kl = jnp.einsum("JDLliad,adub->JDLiubl", Kl, mpo[j+1])
        Klist.append(Kij)

    KVlist = []
    for i in range(n):
        vi = tv[i]
        if len(vi) == 0:
            continue
        Kij = []
        for j in range(n):
            vj = tv[j]
            if len(vj) == 0:
                continue
            if i <= j:
                va = jnp.conj(vi)
                vb = vj
                K = Klist[i][j]
            else:
                va = vi
                vb = jnp.conj(vj)
                K = jnp.einsum("jdliuk->iukjdl", jnp.conj(Klist[i][j]))
            K = jnp.einsum("jdliuk,jda->aliuk", K, va)
            K = jnp.einsum("aliuk,iub->albk", K, vb)
            dims = K.shape
            K = jnp.reshape(K, [dims[0]*dims[1], -1])
            Kij.append(K)
        KVlist.append(Kij)
    return jnp.block(KVlist)


def tangent_vectors(mps):
    # We assume that the state is in the right canonical form
    n = len(mps)
    eps = 1e-10
    lv = []
    tv = []
    r = jnp.array([[1]])
    for i in range(n):
        A = jnp.einsum("ij,jkl->ikl", r, mps[i])
        dims = A.shape
        A = jnp.reshape(A, [-1, dims[-1]])
        u, s, v = jnp.linalg.svd(A, full_matrices=True)
        m = dims[-1]
        mp = sum(s > eps)
        s = s[:m]
        v = v[:m]
        Al = jnp.reshape(u[:, :m], [dims[0], dims[1], m])
        lv.append(Al)
        if m < len(u):
            Ap = jnp.reshape(u[:, mp:], [dims[0], dims[1], -1])
        else:
            Ap = []
        tv.append(Ap)
        r = jnp.einsum("i,ij->ij", s, v)
    return lv, tv


def linearised_specter(mps, mpo0, mpo1, Hright0, Hright1, lamb):
    lv, tv = tangent_vectors(mps)
    K0 = linearised_hamiltonian(mps, lv, tv, mpo0, Hright0)
    K1 = linearised_hamiltonian(mps, lv, tv, mpo1, Hright1)
    a = - max([1 - lamb, 0.])
    b = min([lamb, 1.])
    K = a*K0 + b*K1
    val = jnp.linalg.eigvalsh(K)
    return val, K


def hamiltonian_from_mpo(mpo):
    n = len(mpo)
    H = jnp.array([1.0])
    for i in range(n):
        H = jnp.einsum("...i,ijkl->...jkl", H, mpo[i])
    H = jnp.squeeze(H)
    dims = [2*i for i in range(n)] + [2*i+1 for i in range(n)]
    H = jnp.transpose(H, dims)
    return jnp.reshape(H, [2**n, 2**n])
