import argparse
import numpy as np
from tqdm import tqdm
import os
import time
from GracefulKiller import GracefulKiller

from tdvp_qa.generator import generate_graph, export_graphs, transverse_mpo, longitudinal_mpo
from tdvp_qa.model import PrepareTDVP, export_tdvp_data, generate_postfix
from tdvp_qa.adaptive_model import TDVP_QA
from tdvp_qa.mps import initial_state


def generate_tdvp_filenames(N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, dtr, dti, slope):
    if global_path is None:
        global_path = os.getcwd()
    path_mps = os.path.join(global_path, 'mps/')
    if not os.path.exists(path_mps):
        os.makedirs(path_mps)
    path_data = os.path.join(global_path, 'evolution_data/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = generate_postfix(
        REGULAR, N_verts, N_edges, d, seed, no_local_fields)
    postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_s_{slope}"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    filename_mps = os.path.join(path_mps, 'mps'+postfix+'.hdf5')
    return filename_data, filename_mps


if __name__ == "__main__":
    tstart = time.time()
    killer = GracefulKiller()

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
                        default=0.05,
                        help='Real time step for the simulation.')
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
    parser.add_argument('--dti',
                        type=float,
                        default=0.05,
                        help='Imaginary time step for the simulation.')
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
    parser.add_argument('--no_local_fields',
                        action='store_true',
                        help='If set we set all hi to zero.')
    parser.add_argument('--recalculate',
                        action='store_true',
                        help='We restart the simulation and overwrite existing data.')

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
    # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    d = args_dict['d']
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm
    seed = args_dict['seed']
    no_local_fields = args_dict['no_local_fields']

    if seed is None:
        seed = np.random.randint(10000)
        print(f"Using a random seed {seed}.")

    # TDVP annealing parameters
    stochastic = args_dict["stochastic"]
    adaptive = args_dict["adaptive"]
    slope = args_dict["slope"]
    dtr = args_dict["dtr"]
    dti = args_dict["dti"]
    dt = dtr + 1j*dti
    n = N_verts

    Dmax = args_dict["dmax"]
    recalculate = args_dict["recalculate"]

    Jz, loc_fields, connect = generate_graph(
        N_verts, N_edges, seed=seed, REGULAR=REGULAR, d=d, no_local_fields=no_local_fields, global_path=global_path, recalculate=recalculate)

    assert connect != 0,  "Zero connectivity graph: it corresponds to two isolated subgraphs. The graph will not be saved and the solution will not be computed."

    hz = loc_fields[:, 1]

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    filename_data, filename_mps = generate_tdvp_filenames(
        N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, dtr, dti, slope)

    mpoz = transverse_mpo(Jz, hz, n)
    mpox = longitudinal_mpo(n)

    tensors = initial_state(n, Dmax)

    tdvpqa = TDVP_QA(mpox, mpoz, tensors, slope, dt,
                     compute_states=False, adaptive=adaptive, stochastic=stochastic)

    energies, energiesr, entropies, slopes, states = tdvpqa.evolve(
        evolve_final=True)

    data = {"energy": energies, "energyr": energiesr,
            "entropy": entropies, "slopes": slopes, "states": states}

    export_tdvp_data(data, tdvpqa.mps, filename_data, filename_mps)
    export_graphs(Jz, loc_fields, N_verts, N_edges, seed,
                  connect, REGULAR, d, no_local_fields, global_path)