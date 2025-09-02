import os
import argparse
import numpy as np
import warnings
from tqdm import tqdm
from GracefulKiller import GracefulKiller
from time import time

from tdvp_qa.generator import generate_postfix, TFIM

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general

warnings.filterwarnings("ignore")

LOCAL_PATH = "tfim/exact"


def drive_field(t):
    return t


def tfim_hamiltonian(nx, ny, hx, hz, J=-1, permute=False, seed0=11):
    n = nx*ny
    assert n == len(
        hz), f"The transverse field has size {len(hz)} but should have size n={n}."
    assert n == len(
        hx), f"The longitudinal field has size {len(hx)} but should have size n={n}."

    permutation = None
    if permute:
        permutation = np.random.default_rng(seed0).permutation(n)
        print("Using permutation:", permutation)
    Jz, _, _, permutation = TFIM(
        nx, ny, J, hx, hz, permutation=permutation)

    # local fields
    coeffsx = [[hx[i], i] for i in range(n)]
    coeffsz = [[hz[i], i] for i in range(n) if hz[i] != 0.0]

    # interaction part
    coeffsJz = []
    for edge in Jz:
        i = int(edge[0])
        j = int(edge[1])
        Jij = edge[2]
        coeffsJz.append([Jij, i, j])

    # We increase the strength of the fields linearly in time and keep the interaction part constant (H0).
    # H(t) = H0 + drive_field(t)*H1
    static = [["zz", coeffsJz]]
    dynamic = [["x", coeffsx, drive_field, ()], [
        "z", coeffsz, drive_field, ()]]

    basis = spin_basis_general(n, pauli=-1)

    H = hamiltonian(static, dynamic, basis=basis,
                    dtype=np.float64, check_symm=False, check_herm=False)

    return H


def get_simulation_data(filename_path, ds):
    energies, states, times = [], [], []
    s0 = 0
    if os.path.exists(filename_path):
        try:
            # try loading the simulation data
            data = np.load(filename_path)
            energies = list(data['energies'])
            states = list(data['states'])
            times = list(data['times'])
            s0 = times[-1] + ds
        except:
            print("Corrupted data starting at t=0!")
            pass
    return energies, states, times, s0


def generate_pqc_filename(nx, ny, hx, hz, global_path, ds, seed=11, permute=False):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'tfim/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"nx_{nx}_ny_{ny}_hx_{hx}_hz_{hz}"

    postfix += f"_ds_{ds}"

    if permute:
        postfix += f"_perm_{seed}"

    filename_data = os.path.join(path_data, postfix+'.npz')

    return filename_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=2,
                        help='Width of the lattice.')
    parser.add_argument('--ny', type=int, default=2,
                        help='Height of the lattice.')
    parser.add_argument('--seed', type=int, default=11, help='Random seed.')
    parser.add_argument('--ds', type=float, default=0.01,
                        help='Optimizer step size.')
    parser.add_argument('--path', type=str, default="data/",
                        help='Folder where the simulation data will be saved.')
    parser.add_argument('--hx',
                        type=float,
                        default=-1.5,
                        help='Final field in the x direction. -1.5 (default)')
    parser.add_argument('--hz',
                        type=float,
                        default=0.0,
                        help='Final field in the z direction. 0.0 (default)')
    parser.add_argument('--permute',
                        action='store_true',
                        help='If set we permute the qubits randomly.')
    parser.add_argument('--save_all',
                        action='store_true',
                        help='If set we save all ground states, otherwise save only the last ground state at s=1')

    args = parser.parse_args()

    tstart = time()
    killer = GracefulKiller()

    n = args.nx*args.ny
    ds = args.ds

    filename = generate_pqc_filename(
        nx=args.nx, ny=args.ny, hx=args.hx, hz=args.hz, global_path=args.path, ds=ds, seed=args.seed, permute=args.permute)

    hx = [args.hx]*n
    hz = [args.hz]*n
    H = tfim_hamiltonian(
        args.nx, args.ny, hx, hz, permute=args.permute, seed0=args.seed)

    energies, states, times, s0 = get_simulation_data(filename, ds)

    if s0 > 1:
        print(f"Already done: {filename}")
        return

    for s in tqdm(np.arange(s0, 1+ds, ds)):

        E_GS, psi_GS = H.eigsh(k=2, time=s, which="SA")

        energies.append(E_GS)
        if args.save_all:
            states.append(psi_GS[:, 0])
        times.append(s)

        tcurrent = time()
        if tcurrent-tstart > 3600*45 or killer.kill_now:
            print(
                f"Killing program after {int(tcurrent-tstart)} seconds.")
            break

    if not args.save_all:
        states = [psi_GS]
    np.savez(filename, energies=np.array(energies),
             states=np.array(states), times=np.array(times))

    print(f"Done {n}, {args.seed}, {ds}, {filename}.")


if __name__ == "__main__":
    main()
