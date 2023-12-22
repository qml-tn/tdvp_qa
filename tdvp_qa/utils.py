from jax import jit
import jax.numpy as jnp
from numpy import max, min


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
