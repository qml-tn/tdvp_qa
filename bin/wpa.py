import argparse
import numpy as np
import os
import pickle

from tdvp_qa.generator import generate_graph, export_graphs, transverse_mpo, longitudinal_mpo
from tdvp_qa.model import generate_postfix
from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.mps import initial_state_theta

from jax import config


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


def generate_tdvp_filename(n, seed, global_path, annealing_schedule, Dmax, dtr,
                           dti, slope, seed_tdvp, stochastic, double_precision, slope_omega, rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'wpa/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"n_{n}_s_{seed}"

    if nitime > 1:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_{nitime}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"
    else:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"

    if rand_xy:
        postfix += "_xy"
    if scale_gap:
        postfix += "_sgap"
    if auto_grad:
        postfix += "_ag"
    if cyclic_path:
        postfix += "_cycle"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    return filename_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--model_folder',
                        default="/home/bojanz/projekti/variational_annealing/piqmc/data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--regular',
                        action='store_true',
                        help='If set we generate regular graph of degree d.')
    parser.add_argument('--n',
                        type=int,
                        default=32,
                        help='Number of spins.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.25,
                        help='Wishart planted ensemble parameter which specifies the number-of-equations–to–number-of-variables ratio. 0.25 (default)')
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
                        default=0.001,
                        help='Initial increment for which we change lambda after each time step.')
    parser.add_argument('--slope_omega',
                        type=float,
                        default=0.001,
                        help='The ratio between the slope and omega during the adaptive time evolution.')
    parser.add_argument('--seed',
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

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    n = args_dict['n']
    alpha = args_dict['alpha']

    # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm
    seed = args_dict['seed']

    if seed is None:
        seed = np.random.randint(10000)
        print(f"Using a random seed {seed}.")

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

    cyclic_path = args_dict["cyclic_path"]

    if dti > 0:
        nitime = args_dict["nitime"]
    else:
        nitime = 0

    rand_init = args_dict["rand_init"]
    rand_xy = args_dict["rand_xy"]
    auto_grad = args_dict["auto_grad"]

    Dmax = args_dict["dmax"]
    recalculate = args_dict["recalculate"]
    compute_states = args_dict["comp_state"]

    if double_precision:
        config.update("jax_enable_x64", True)

    model_folder = args_dict["model_folder"]

    model_name = f"wpe_size{n}_alpha{alpha}_realization{seed}.txt"
    model_path = os.path.join(model_folder, f"wishart_N{n}", model_name)
    print(model_path)
    if os.path.exists(model_path):
        Jz_matrix = np.loadtxt(model_path)
        # the planted state is a ferromagnetic state with all spins up!
        E0 = -4*np.sum(Jz_matrix)
        Jz = []
        for i in range(n):
            # Jz.append([i, i, Jz_matrix[i, i]]) # There are no local interactions by definition
            for j in range(i):
                Jz.append([j, i, -8*Jz_matrix[j, i]])
        Jz = np.array(Jz)
    else:
        Exception(
            "No model with this parameters generated! Implement the generator")

    hz = np.zeros(n)  # The fields are always zero in this model

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    theta = np.array([[np.pi/2., 0]]*n)
    if rand_init:
        np.random.seed(seed_tdvp)
        theta = np.array(
            [[np.random.rand()*0.1+np.pi/2, 2*np.random.rand()*np.pi] for i in range(n)])
        if rand_xy:
            theta[:, 0] = np.pi/2.

    filename = generate_tdvp_filename(n, seed, global_path, annealing_schedule, Dmax, dtr,
                                      dti, slope, seed_tdvp, stochastic, double_precision, slope_omega, rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path)

    mpox = longitudinal_mpo(n, theta)
    mpoz = transverse_mpo(Jz, hz, n)
    tensors = initial_state_theta(n, Dmax, theta=theta)

    if recalculate:
        data = None
    else:
        data = get_simulation_data(filename)

    lamb = 0
    if data is not None:
        tensors = data["mps"]
        slope = data["slope"][-1]
        lamb = np.sum(data["slope"])

    if lamb < 1:
        tdvpqa = TDVP_QA_V2(mpox, mpoz, tensors, slope, dt, lamb=lamb, max_slope=0.05, min_slope=1e-8,
                            adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega,
                            ds=0.01, scale_gap=scale_gap, auto_grad=auto_grad, nitime=nitime, cyclic_path=cyclic_path)

        data = tdvpqa.evolve(data=data)
        data["mps"] = [np.array(A) for A in tdvpqa.mps.tensors]
        data["Jz"] = Jz
        data["hz"] = hz

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        # export_graphs(Jz, loc_fields, N_verts, N_edges, seed,
        #               connect, REGULAR, d, no_local_fields, global_path, max_cut)

        print(f"Final energy is: {data['energy'][-1]}")

        if compute_states:
            psi1 = data["var_gs"][-1]
        else:
            psi1 = data["state"][-1]

        for ps in data["state"][-1]:
            print(abs(ps[0, :, 0]), np.linalg.norm(ps[0, :, 0]))

        sol = np.array(
            [2*int(abs(ps[0, 0, 0]) > abs(ps[0, 1, 0]))-1 for ps in psi1])

        print(sol)
        print(f"Var_gs energy: {4*sol @ Jz_matrix @ sol}")
        print(f"Ground state energy: {E0}")
        s_up = np.ones(n)
        print(f"Residual energy state energy: {4*(sol @ Jz_matrix @ sol - s_up @ Jz_matrix @ s_up )/abs(E0)}")
    else:
        print("The simulation is already finished!")
