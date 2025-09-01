import argparse
import numpy as np
import os
import pickle

from tdvp_qa.generator import Wishart, transverse_mpo, longitudinal_mpo, flat_sx_H0, TFIM, ising_with_field_mpo
from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.mps import initial_state_theta, operator_profile

from jax import config
import jax.numpy as jnp


def get_simulation_data(filename_path):
    data = None
    if os.path.exists(filename_path):
        try:
            with open(filename_path, 'rb') as f:
                data = pickle.load(f)
        except:
            print("Corrupted data starting at t=0!")
            pass
    return data


def generate_tdvp_filename(nx, ny, hx, hz, global_path, annealing_schedule, Dmax, dtr,
                           dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                           rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, seed0=None, permute=False, sin_lambda=False, T=None):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'tfim/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"nx_{nx}_ny_{ny}_hx_{hx}_hz_{hz}"

    if T is not None:
        postfix += f"_{annealing_schedule}_D_{Dmax}_T_{T}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"
    elif nitime > 1:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_{nitime}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"
    else:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"

    if rand_init:
        postfix += f"_{seed0}"

    if permute:
        postfix += f"_perm_{seed0}"
    if rand_xy:
        postfix += "_xy"
    if scale_gap:
        postfix += "_sgap"
    if auto_grad:
        postfix += "_ag"
    if cyclic_path:
        postfix += "_cycle"
    if sin_lambda:
        postfix += "_sin"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    return filename_data


def gs_overlap(tensors, s0, sol):
    overlap = jnp.array([1.0])
    for A, s in zip(tensors, sol):
        overlap = jnp.einsum("i,ij->j", overlap, A[:, int(np.mod(s+s0, 2)), :])
    overlap = jnp.abs(overlap[0])
    overlap = float(overlap)
    # print("overlap", overlap)
    return overlap


def data_callback(data, tensors):
    if "mx" not in data.keys():
        data["mx"] = []
    if "mz" not in data.keys():
        data["mz"] = []
    data["mx"].append(operator_profile(jnp.array([[0, 1], [1, 0]]), tensors))
    data["mz"].append(operator_profile(jnp.array([[1, 0], [0, -1]]), tensors))
    return data


def coupling_fn(lamb):
    return 1, lamb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--regular',
                        action='store_true',
                        help='If set we generate regular graph of degree d.')
    parser.add_argument('--nx',
                        type=int,
                        default=2,
                        help='Length of the lattice.')
    parser.add_argument('--ny',
                        type=int,
                        default=2,
                        help='Width of the lattice.')
    parser.add_argument('--J',
                        type=float,
                        default=-1.0,
                        help='Strength of the nearest neighbour coupling (default) -1.0.')
    parser.add_argument('--hx',
                        type=float,
                        default=-1.5,
                        help='Final field in the x direction. -1.5 (default)')
    parser.add_argument('--hz',
                        type=float,
                        default=0.0,
                        help='Final field in the z direction. 0.0 (default)')
    parser.add_argument('--dmax',
                        type=int,
                        default=8,
                        help='Maximum bond dimension to be used in the TDVP.')
    parser.add_argument('--dtr',
                        type=float,
                        default=0.025,
                        help='Real time step for the simulation.')
    parser.add_argument('--dti',
                        type=float,
                        default=0.0,
                        help='Imaginary time step for the simulation.')
    parser.add_argument('--nitime',
                        type=int,
                        default=1,
                        help='Number of repetitions of the same step if dti>0.')
    parser.add_argument('--stochastic',
                        action='store_true',
                        help='If set we sample the imaginary component of dt in the range [0, dti].')
    parser.add_argument('--adaptive',
                        action='store_true',
                        help='If set we adaptively change the slope.')
    parser.add_argument('--slope',
                        type=float,
                        default=0.01,
                        help='Initial increment for which we change lambda after each time step.')
    parser.add_argument('--slope_omega',
                        type=float,
                        default=0.01,
                        help='The ratio between the slope and omega during the adaptive time evolution.')
    parser.add_argument('--seed',
                        type=int,
                        action="store",
                        help='Seed for the graph generator.')
    parser.add_argument('--seed0',
                        type=int,
                        action="store",
                        help='Seed for the graph generator.')
    parser.add_argument('--seed_tdvp',
                        type=int,
                        default=42,
                        help='Seed for the mps and tdvp evolution. Used for sampling and stochastic TDVP.')
    parser.add_argument('--recalculate',
                        action='store_true',
                        help='We restart the simulation and overwrite existing data.')
    parser.add_argument('--double_precision',
                        action='store_true',
                        help='If set we use double precision jax calculations.')
    parser.add_argument('--comp_state',
                        action='store_true',
                        help='If set we also compute the states.')
    parser.add_argument('--rand_init',
                        action='store_true',
                        help='If set the initial state will be a Haar random product state. The initial hamiltonian is changed accordingly.')
    parser.add_argument('--rand_xy',
                        action='store_true',
                        help='If set the initial state will be a random product state in the XY plane, i.e. sz=0. The initial hamiltonian is changed accordingly.')
    parser.add_argument('--scale_gap',
                        action='store_true',
                        help='If set we use gap scaling. This can be used with or without adaptive step adjustment.')
    parser.add_argument('--auto_grad',
                        action='store_true',
                        help='If set we automatic_gradient.')
    parser.add_argument('--cyclic_path',
                        action='store_true',
                        help='If set we make a cyclic path and the final point is the same as the initial point.')
    parser.add_argument('--checkpoint',
                        action='store_true',
                        help='If set we save the solution at every 0.01 increase in s.')
    parser.add_argument('--sin_lambda',
                        action='store_true',
                        help='If set we use cos(0.5*lambda*pi) and sin(0.5*lambda*pi) prefactors of H0 and H1.')
    parser.add_argument('--permute',
                        action='store_true',
                        help='If set we permute Jz and hz in the final Hamiltonian.')
    parser.add_argument('--max_hours',
                        type=float,
                        default=40,
                        help='Maximum number of hours after which the simulation of the evolution will stop.')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='If set we shuffle the ground state to be a random strin not a fully polarised one.')
    parser.add_argument('--T',
                        type=float,
                        action="store",
                        help='If set we use use a Monte Carlo sampling with the temperature T of the eigenstates of the Heff instead of the real/imaginary time evolution.')

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    nx = args_dict['nx']
    ny = args_dict['ny']
    hx = args_dict['hx']
    hz = args_dict['hz']
    J = args_dict["J"]
    n = nx*ny

    # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm
    seed = args_dict['seed']

    if seed is None:
        seed = np.random.randint(10000)
        print(f"Using a random seed {seed}.")

    seed0 = args_dict["seed0"]
    shuffle = args_dict["shuffle"]
    Tmc = args_dict["T"]

    if seed0 is None:
        seed0 = np.random.randint(10000)
        print(f"Using a initial random seed {seed0}.")

    # TDVP annealing parameters
    double_precision = args_dict["double_precision"]
    seed_tdvp = args_dict["seed_tdvp"]
    stochastic = args_dict["stochastic"]
    adaptive = args_dict["adaptive"]
    scale_gap = args_dict["scale_gap"]
    slope_omega = args_dict["slope_omega"]
    slope = args_dict["slope"]
    lamb = 0
    dtr = args_dict["dtr"]
    dti = args_dict["dti"]
    dt = dtr - 1j*dti
    auto_grad = args_dict["auto_grad"]

    cyclic_path = args_dict["cyclic_path"]

    if dti > 0 or auto_grad:
        nitime = args_dict["nitime"]
    else:
        nitime = 0

    rand_init = args_dict["rand_init"]
    rand_xy = args_dict["rand_xy"]

    Dmax = args_dict["dmax"]
    recalculate = args_dict["recalculate"]
    compute_states = args_dict["comp_state"]
    checkpoint = args_dict["checkpoint"]
    permute = args_dict["permute"]
    sin_lambda = args_dict["sin_lambda"]

    max_training_hours = args_dict["max_hours"]

    if double_precision:
        config.update("jax_enable_x64", True)

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    theta = np.array([[np.pi/2., 0]]*n)
    if rand_init:
        # np.random.seed(seed0)
        rng = np.random.default_rng(seed0)
        # theta = np.array([[rng.uniform(-1,1)*0.1+np.pi/2, 2*rng.uniform()*np.pi] for i in range(n)])
        # print(theta[:,0])
        theta = np.array(
            [[rng.uniform()*np.pi, 2*rng.uniform()*np.pi] for i in range(n)])

        if rand_xy:
            theta[:, 0] = np.pi/2.

    filename = generate_tdvp_filename(nx, ny, hx, hz, global_path, annealing_schedule, Dmax, dtr,
                                      dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                                      rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, seed0, sin_lambda=sin_lambda, permute=permute, T=Tmc)

    if recalculate:
        data = None
    else:
        data = get_simulation_data(filename)

    if data is None:
        permutation = None
        if permute:
            permutation = np.random.default_rng(seed0).permutation(n)
            print("Using permutation:", permutation)
        Jz, _, _, permutation = TFIM(
            nx, ny, J, hx, hz, permutation=permutation)
    else:
        Jz = data["Jz"]
        hz = data["hz"]
        hx = data["hx"]
        J = data["J"]
        permutation = data["permutation"] if "permutation" in data.keys(
        ) else None

    # mpo1 = ising_with_field_mpo(Jz, hz, hx, n)
    mpo1 = longitudinal_mpo(n, [None]*n, g=hx)
    mpo0 = ising_with_field_mpo(Jz, np.zeros(n), np.zeros(n), n)
    assert J < 0, "The coupling J should be negative for the TFIM initial Hamiltonian."

    theta = np.array([[0.0, 0.0]]*n)
    tensors = initial_state_theta(n, Dmax, theta=theta)

    lamb = 0
    if data is not None:
        tensors = data["mps"]
        slope = data["slope"][-1]
        lamb = np.sum(data["slope"])

    if lamb < 1:
        tdvpqa = TDVP_QA_V2(mpo0, mpo1, tensors, slope, dt, lamb=lamb, max_slope=0.05, min_slope=1e-8,
                            adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega,
                            ds=0.01, scale_gap=scale_gap, auto_grad=auto_grad, nitime=nitime, cyclic_path=cyclic_path, sin_lambda=sin_lambda, Tmc=Tmc, coupling_fn=coupling_fn)

        data = tdvpqa.evolve(data=data,  filename=filename,
                             checkpoint=checkpoint, max_training_hours=max_training_hours, data_callback=data_callback)
        data["mps"] = [np.array(A) for A in tdvpqa.mps.tensors]
        data["Jz"] = Jz
        data["J"] = J
        data["hx"] = hx
        data["hz"] = hz
        data["permutation"] = permutation

        print(f"Saving: {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Final energy is: {data['energy'][-1]}")
        print("Done!")
    else:
        print("The simulation is already finished!")
