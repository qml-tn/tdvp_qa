from jax import jit
import jax.numpy as jnp
from numpy import max, min, prod


def annealing_energy_canonical(Hl0, Hl1, Hr0, Hr1, H0, H1, lamb, A):
    Dl, d, Dr = A.shape
    a = -max([1 - lamb, 0.])
    b = min([lamb, 1.])

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


@jit
def effective_hamiltonian_A(Hl, Hr, H0):
    Heff = jnp.einsum("umd,mijn->diujn", Hl, H0)
    Heff = jnp.einsum("diujn,anb->dibuja", Heff, Hr)
    return Heff


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


def linearised_hamiltonian(mps, mpo, Hright):
    # We assume that the mps is in the right canonical form
    n = len(mps)
    Hl = jnp.array([[[1.]]])
    dims = [prod(A.shape) for A in mps]
    # N = sum(dims)
    Klist = []
    A1 = mps[0]
    for i in range(n):
        # Diagonal of K
        Kl = jnp.einsum("iaj,adub->jdiub", Hl, mpo[i])
        K = jnp.einsum("jdiub,kbl->jdliuk", Kl, Hright[i])
        dl = dims[i]
        dr = dl
        Kij = []
        for j in range(i):
            Kij.append(jnp.conj(jnp.transpose(Klist[j][i])))
        Kij.append(jnp.reshape(K, [dl, dr]))
        # Move Hl and mps to the right
        A = jnp.reshape(A1, [-1, A1.shape[-1]])
        q, r = jnp.linalg.qr(A)
        A = jnp.reshape(q, A1.shape)
        Hl = left_hamiltonian(A, Hl, mpo[i])
        # Preparing first off-diagonal Kl
        Kl = jnp.einsum("jdiub,iuk->jdkb", Kl, A)
        if i < n-1:
            A1 = jnp.einsum("ij,jkl->ikl", r, mps[i+1])
            A2 = A1.copy()
            Kl = jnp.einsum("JDia,adub->JDiudb", Kl, mpo[i+1])
            Kl = jnp.einsum("JDiudb,Ldl->JDLiubl", Kl, jnp.conj(mps[i+1]))

        # Calculate offdiagonal elements of K
        for j in range(i+1, n):
            # Calculating the offdiagonal K
            K = jnp.einsum("JDLiubl,kbl->JDLiuk", Kl, Hright[j])
            dr = dims[j]
            Kij.append(jnp.reshape(K, [dl, dr]))

            # Calculating the new Kl and moving A2 to right
            A = jnp.reshape(A2, [-1, A2.shape[-1]])
            q, r = jnp.linalg.qr(A)
            A = jnp.reshape(q, A2.shape)
            if j < n-1:
                A2 = jnp.einsum("ij,jkl->ikl", r, mps[j+1])
                Kl = jnp.einsum("JDLkual,kui->JDLial", Kl, A)
                Kl = jnp.einsum("JDLiar,rdl->JDLliad", Kl, jnp.conj(mps[j+1]))
                Kl = jnp.einsum("JDLliad,adub->JDLiubl", Kl, mpo[j+1])
        Klist.append(Kij)
    return jnp.block(Klist)


def linearised_specter(mps, mpo0, mpo1, Hright0, Hright1, lamb):
    K0 = linearised_hamiltonian(mps, mpo0, Hright0)
    K1 = linearised_hamiltonian(mps, mpo1, Hright1)
    K = K0 + lamb * K1
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
