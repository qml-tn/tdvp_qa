import argparse
import numpy as np
import os
import pickle

from tdvp_qa.generator import generate_graph, export_graphs, Wishart, transverse_mpo, longitudinal_mpo, flat_sx_H0
from tdvp_qa.model import generate_postfix
from tdvp_qa.annealing_catalyst import TDVP_MULTI_MPS
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


def generate_tdvp_filename(N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, dtr,
                           dti, slope, seed_tdvp, stochastic, double_precision, slope_omega, rand_init, rand_xy, scale_gap,
                           max_cut, auto_grad, nitime, cyclic_path, inith, nmps=1, reorder_mps=False, seed0=None, alpha0=None, cat_strength=4):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'adaptive_data_v2/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = generate_postfix(
        REGULAR, N_verts, N_edges, d, seed, no_local_fields, max_cut=max_cut)

    if nitime > 1:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_{nitime}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"
    else:
        postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"

    if rand_init:
        postfix += f"_{seed0}"

    postfix += f"_h0_{inith}"
    if inith == "wishart":
        postfix += f"_{seed0}_{alpha0}"

    if rand_xy:
        postfix += "_xy"
    if scale_gap:
        postfix += "_sgap"
    if auto_grad:
        postfix += "_ag"
    if cyclic_path:
        postfix += "_cycle"
    if nmps > 1:
        postfix += f"_nmps_{nmps}_{cat_strength}"
        if reorder_mps:
            postfix += "_rm"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    return filename_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--regular',
                        action='store_true',
                        help='If set we generate regular graph of degree d.')
    parser.add_argument('--n_verts',
                        type=int,
                        default=16,
                        help='Number of vertices in the graph.')
    parser.add_argument('--n_edges',
                        type=int,
                        action="store",
                        help='Number of edges.')
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
    parser.add_argument('--distr',
                        type=str,
                        default="Uniform",
                        help='Sampling distribution of the local fields. It can be Normal or Uniform')
    parser.add_argument('--d',
                        type=int,
                        action="store",
                        help='Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten.')
    parser.add_argument('--seed',
                        type=int,
                        action="store",
                        help='Seed for the graph generator.')
    parser.add_argument('--seed_tdvp',
                        type=int,
                        default=42,
                        help='Seed for the mps and tdvp evolution. Used for sampling and stochastic TDVP.')
    parser.add_argument('--no_local_fields',
                        action='store_true',
                        help='If set we set all hi to zero.')
    parser.add_argument('--inith',
                        default="sx",
                        type=str,
                        help='Choice of the initial Hamiltonian. sx (default), flatsx, wishart')
    parser.add_argument('--max_cut',
                        action='store_true',
                        help='If set to true a max-cut hamiltonian is produced and local fields are set to zero.')
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
    parser.add_argument('--seed0',
                        type=int,
                        action="store",
                        help='Seed for the graph generator.')
    parser.add_argument('--alpha0',
                        type=float,
                        default=0.9,
                        help='Initial Wishart planted ensemble parameter which specifies the number-of-equations–to–number-of-variables ratio. 0.25 (default)')
    parser.add_argument('--cat_strength',
                        type=float,
                        default=4.0,
                        help='Maximum strength of the catalyst hamiltonian.')
    parser.add_argument('--reorder_mps',
                        action='store_true',
                        help='If set we reorder the MPS with respect to the energy after each step.')
    parser.add_argument('--nmps',
                        type=int,
                        default=3,
                        help='Number of total MPS used for the evolution.')

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    N_verts = args_dict['n_verts']
    N_edges = args_dict['n_edges']
    if N_edges is None:
        N_edges = int(N_verts*(N_verts - 1)/2)

    # Set TRUE to generate regular graph of degree d
    REGULAR = args_dict['regular']
    if REGULAR:
        N_edges = None
    d = args_dict['d']

    # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm
    seed = args_dict['seed']
    no_local_fields = args_dict['no_local_fields']
    inith = args_dict["inith"]

    max_cut = args_dict["max_cut"]
    if max_cut:
        no_local_fields = True

    if seed is None:
        seed = np.random.randint(10000)
        print(f"Using a random seed {seed}.")

    seed0 = args_dict["seed0"]
    alpha0 = args_dict["alpha0"]

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
    n = N_verts

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

    nmps = args_dict["nmps"]
    reorder_mps = args_dict["reorder_mps"]
    cat_strength = args_dict["cat_strength"]

    Jz, loc_fields, connect = generate_graph(
        N_verts, N_edges, seed=seed, REGULAR=REGULAR, d=d, no_local_fields=no_local_fields, global_path=global_path, recalculate=recalculate, max_cut=max_cut)

    assert connect != 0,  "Zero connectivity graph: it corresponds to two isolated subgraphs. The graph will not be saved and the solution will not be computed."

    hz = loc_fields[:, 1]

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    theta = np.array([[np.pi/2., 0]]*n)
    if rand_init:
        # np.random.seed(seed_tdvp)
        # theta = np.array([[np.random.rand()*np.pi, 2*np.random.rand()*np.pi] for i in range(n)])
        rng = np.random.default_rng(seed0)
        theta = np.array(
            [[rng.uniform()*np.pi, 2*rng.uniform()*np.pi] for i in range(n)])
        if rand_xy:
            theta[:, 0] = np.pi/2.

    filename = generate_tdvp_filename(
        N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, dtr, dti, slope, seed_tdvp=seed_tdvp, stochastic=stochastic,
        double_precision=double_precision, slope_omega=slope_omega, rand_init=rand_init, rand_xy=rand_xy, scale_gap=scale_gap, max_cut=max_cut, auto_grad=auto_grad,
        nitime=nitime, cyclic_path=cyclic_path, inith=inith, alpha0=alpha0, seed0=seed0, nmps=nmps, reorder_mps=reorder_mps, cat_strength=cat_strength)

    if inith == "flatsx":
        mpox = flat_sx_H0(n)
    elif inith == "sx":
        mpox = longitudinal_mpo(n, theta)
    elif inith == "wishart":
        Jx, hx, _ = Wishart(n, alpha=alpha0, seed=seed0)
        # We have to reverse the sign of Jx since the first hamiltonian comes with the - sign in front
        Jx[:, -1] = -Jx[:, -1]
        mpox = transverse_mpo(Jx, hx, n, rotate_to_x=True)
    else:
        mpox = longitudinal_mpo(n, theta)

    mpoz = transverse_mpo(Jz, hz, n)

    if recalculate:
        data = None
    else:
        data = get_simulation_data(filename)

    tensorslist = [initial_state_theta(
        n, Dmax, theta=theta) for _ in range(nmps)]

    lamb = 0
    if data is not None:
        tensorslist = data["mpslist"]
        slope = data["slope"][-1]
        lamb = np.sum(data["slope"])

    if lamb < 1:
        tdvpqa = TDVP_MULTI_MPS(mpox, mpoz, tensorslist, slope, dt, lamb=lamb, max_slope=0.05, min_slope=1e-8,
                                adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega,
                                ds=0.01, scale_gap=scale_gap, auto_grad=auto_grad, nitime=nitime, cyclic_path=cyclic_path,
                                Tmc=None, nmps=nmps, reorder_mps=reorder_mps, cat_strength=cat_strength)

        data = tdvpqa.evolve(data=data)

        data["Jz"] = Jz
        data["hz"] = hz

        data["mpslist"] = [[np.array(A) for A in mps.tensors]
                           for mps in tdvpqa.mpslist]

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved simulation data to {filename}")

        export_graphs(Jz, loc_fields, N_verts, N_edges, seed,
                      connect, REGULAR, d, no_local_fields, global_path, max_cut)
        if max_cut:
            mcut = (np.sum(Jz[:, 2])-data["energy"][-1])/2
            print(f"Edges in maximum cut: {mcut}")
        else:
            print(f"Final energies are: {data['energy'][-1]}")
    else:
        print("The simulation is already finished!")
        print(f"The data is stored in {filename}")
