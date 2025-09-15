import argparse
import numpy as np
import os
import pickle

from tdvp_qa.generator import transverse_mpo, longitudinal_mpo
from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.mps import initial_state_theta

from jax import config
import json

from tqdm import tqdm


def get_instance_data(filename):
    with open(filename, 'r') as f_json:
        # Load QUBO data from JSON file
        qubo_data = json.load(f_json)
        n = qubo_data['qubo']['shape'][0]
        A = np.zeros((n, n))
        J = []
        for row, col, val in zip(qubo_data['qubo']['row'], qubo_data['qubo']['col'], qubo_data['qubo']['data']):
            A[row, col] = val
            if row != col:
                i = np.min([row, col])
                j = np.max([row, col])
                # Divide by 4 to convert to Ising coupling
                J.append([i, j, 0.25*val])
        J = np.array(J)

    # Ensure symmetry
    A = (A + A.T) / 2

    # Constant term
    c = 0.25 * np.sum(A) + np.sum(np.diag(A)/4)

    # Linear terms
    h = 0.5 * np.sum(A, axis=1)

    # Quadratic terms matrix
    # Jmat = 0.25 * A
    # np.fill_diagonal(Jmat, 0.0)  # convention: no self-interaction
    return J, h, c, n, A


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


def generate_tdvp_filename(filename, global_path, annealing_schedule, Dmax, dtr,
                           dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                           rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, seed0=None, permute=False, sin_lambda=False, T=None, scaled=False):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'solutions')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = filename.split('/')[-1].replace('.json', '')

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
    if scaled:
        postfix += "_scaled"

    filename_data = os.path.join(path_data, postfix+'.pkl')
    return filename_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/rudolf/instances/exact/40",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--filename',
                        type=str,
                        help='Local path from to the file.')
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
    parser.add_argument('--scale_energy',
                        action='store_true',
                        help='If set we scale the energy of the final hamiltonian by first calculating the approximate final ground state energy.')
    parser.add_argument('--T',
                        type=float,
                        action="store",
                        help='If set we use use a Monte Carlo sampling with the temperature T of the eigenstates of the Heff instead of the real/imaginary time evolution.')

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Filename of the QUBO instance
    filename = args_dict['filename']

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

    filename = generate_tdvp_filename(filename, global_path, annealing_schedule, Dmax, dtr,
                                      dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                                      rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, seed0, sin_lambda=sin_lambda, permute=permute, T=Tmc, scaled=scaled)

    if recalculate:
        data = None
    else:
        data = get_simulation_data(filename)

    # Load Jz and hz from json file
    if data is None:
        # get_instance_data(filename)
        file_path = os.path.join(global_path, args_dict['filename'])
        Jz, hz, c, n, Aqubo = get_instance_data(file_path)
        scale = 1
    else:
        Jz = data["Jz"]
        hz = data["hz"]
        c = data["c"]
        n = data["n"]
        Aqubo = data["A"]
        scale = data["scale"]

    mpo0 = longitudinal_mpo(n, [None]*n)
    mpo1 = transverse_mpo(Jz, hz, n)

    theta = np.array([[np.pi/2., 0]]*n)
    tensors = initial_state_theta(n, Dmax, theta=theta)

    lamb = 0
    if data is not None:
        tensors = data["mps"]
        slope = data["slope"][-1]
        lamb = np.sum(data["slope"])

    if args_dict["scale_energy"] and lamb == 0 and scale == 1:
        # Scaling the final hamiltonian such that the final energy is approximately extensive!
        # We save the scaling factor to get the correct final energy
        tensors0 = initial_state_theta(n, Dmax=4, theta=theta)
        tdvpqa0 = TDVP_QA_V2(mpo0, mpo1, tensors0, slope=0.001, dt=0.001+1j*0.001, lamb=0, max_slope=0.05, min_slope=1e-8,
                             adaptive=False, compute_states=False, key=seed_tdvp, slope_omega=slope_omega,
                             ds=0.1, auto_grad=True, nitime=1, cyclic_path=False, sin_lambda=True)
        data_scale = tdvpqa0.evolve()
        # Only scale if the energy is not close to 1 to avid problems of division with zero.
        if abs(data_scale["energy"][-1]) > 0.01:
            scale = abs(data_scale["energy"][-1])/n
            print(
                f"The approximate final energy is {data_scale['energy'][-1]}.")
            print(f"Scale factor is {scale}")

        # Scale the final MPO
        Jz[:, 2] = Jz[:, 2]/scale
        hz = hz / scale
        mpo1 = transverse_mpo(Jz, hz, n)

    if lamb < 1:
        tdvpqa = TDVP_QA_V2(mpo0, mpo1, tensors, slope, dt, lamb=lamb, max_slope=0.05, min_slope=1e-8,
                            adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega,
                            ds=0.01, scale_gap=scale_gap, auto_grad=auto_grad, nitime=nitime, cyclic_path=cyclic_path, sin_lambda=sin_lambda, Tmc=Tmc)

        data = tdvpqa.evolve(data=data,  filename=filename,
                             checkpoint=checkpoint, max_training_hours=max_training_hours)
        data["mps"] = [np.array(A) for A in tdvpqa.mps.tensors]
        data["Jz"] = Jz
        data["hz"] = hz
        data["n"] = n
        data["c"] = c
        data["A"] = Aqubo
        data["scale"] = scale

        print(f"Saving: {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Final energy is: {data['energy'][-1]}")
        print(
            f"Corresponding QUBO cost function value is: {scale*data['energy'][-1] + c}")
        print("Done!")

        # Generate samples from the final state
        Nsamp = 100
        final_mps = tdvpqa.mps
        samples = []
        print("Generating samples...")
        best_energy = np.inf
        best_sample = []
        for _ in tqdm(range(Nsamp)):
            samp = 1-np.array(final_mps.sample())
            samples.append(samp)
            energy = np.sum(samp @ Aqubo * samp)
            if energy < best_energy:
                best_energy = energy
                best_sample = samp.copy()

        # saving samples in the JSON format
        samples = np.array(samples)
        solution_list = samples.tolist()
        solution_dict = {"x": list(best_sample),
                         "cost": best_energy, "samples": solution_list}
        with open(filename.replace('.pkl', '.json'), 'w') as f_json:
            json.dump(solution_dict, f_json)
        print(f"Best sample cost {best_energy}")
        print(f"Best sample: {best_sample}")
    else:
        print("The simulation is already finished!")
