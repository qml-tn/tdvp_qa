import numpy as np


def hamiltonian_from_mpo(mpo):
    n = len(mpo)
    H = mpo[0][0]
    inde = [0]
    indo = [1]
    for i in range(1, n):
        inde.append(2*i)
        indo.append(2*i+1)
        if i == n-1:
            H = np.einsum("...l,lij->...ij", H, mpo[i][:, :, :, 0])
        else:
            H = np.einsum("...l,lijr->...ijr", H, mpo[i])
    inds = inde+indo
    H = np.transpose(H, axes=inds)
    n = len(inde)
    d = int(2**n)
    return np.reshape(H, [d, d])

def state_from_mps(tensors):
    n = len(tensors)
    assert n < 25, f"Can not construct a state with more than 25 spins. The state has {n} spins"
    psi = tensors[0]
    for i in range(1, n):
        psi = np.einsum("...i,ijk", psi, tensors[i])
    return np.reshape(psi, [-1])


def Sx(i, n):
    return np.kron(np.eye(2**(i)), np.kron([[0, 1], [1, 0]], np.eye(2**(n-i-1))))


def Sy(i, n):
    return np.kron(np.eye(2**(i)), np.kron([[0, -1j], [1j, 0]], np.eye(2**(n-i-1))))


def Sz(i, n):
    return np.kron(np.eye(2**(i)), np.kron([[1, 0], [0, -1]], np.eye(2**(n-i-1))))


def Ising(Jz, hz):
    n = len(hz)
    H = 0
    for i in range(n):
        H += hz[i]*Sz(i, n)
    for k in range(len(Jz)):
        i = int(Jz[k, 0])
        j = int(Jz[k, 1])
        H += Jz[k, 2]*Sz(i, n)@Sz(j, n)
    return H


def Hx_fields(n,hx=None):
    H = 0
    if hx is None:
        for i in range(n):
            H += Sx(i, n)
    else:
        for i in range(n):
            H += Sx(i, n)*hx[i]
    return H


def energy(sample, Jz, hz):
    conf = 1.-2.*sample
    return (hz@conf + conf @ Jz @ conf)


def energy_mpo(sample, mpo):
    e = np.array([[1.]])
    for i in range(len(sample)):
        s = int(sample[i])
        e = e @ mpo[i][:, s, s, :]
    return e[0, 0]
