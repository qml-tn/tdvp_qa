import argparse
import numpy as np
import os
import pickle
import scipy
from tqdm import tqdm

from tdvp_qa.generator import generate_graph, export_graphs, transverse_mpo, longitudinal_mpo
from tdvp_qa.model import generate_postfix
from tdvp_qa.exact import Ising, Hx_fields

def get_simulation_data(filename_path):
    data = None
    if os.path.exists(filename_path):
        with open(filename_path, 'rb') as f:
            data = pickle.load(f)
    return data


def generate_tdvp_filename(N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, dtr, dti, slope, seed_tdvp, stochastic, slope_omega):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'exact/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = generate_postfix(
        REGULAR, N_verts, N_edges, d, seed, no_local_fields)

    postfix += f"_{annealing_schedule}_dt_{dtr}_{dti}_s_{slope}_s_{stochastic}_{seed_tdvp}_so_{slope_omega}"

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
    parser.add_argument('--dtr',
                        type=float,
                        default=0.025,
                        help='Real time step for the simulation.')
    parser.add_argument('--dti',
                        type=float,
                        default=0.0,
                        help='Imaginary time step for the simulation.')
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
                        default=0.00001,
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
    seed_tdvp = args_dict["seed_tdvp"]
    stochastic = args_dict["stochastic"]
    adaptive = args_dict["adaptive"]
    slope_omega = args_dict["slope_omega"]
    slope = args_dict["slope"]
    dtr = args_dict["dtr"]
    dti = args_dict["dti"]
    dt = dtr - 1j*dti
    n = N_verts

    recalculate = args_dict["recalculate"]


    Jz, loc_fields, connect = generate_graph(
        N_verts, N_edges, seed=seed, REGULAR=REGULAR, d=d, no_local_fields=no_local_fields, global_path=global_path, recalculate=recalculate)

    assert connect != 0,  "Zero connectivity graph: it corresponds to two isolated subgraphs. The graph will not be saved and the solution will not be computed."

    hz = loc_fields[:, 1]

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    filename = generate_tdvp_filename(
        N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, dtr, dti, slope, seed_tdvp=seed_tdvp, stochastic=stochastic, slope_omega=slope_omega)

    H0 = Hx_fields(n)
    H1 = Ising(Jz, hz)

    states_exact = []
    gs_exact = []
    slist = []
    specter = []
    pbar = tqdm(total=1, position=0, leave=True)
    k = 1
    ds = 0.01
    lamb = 0
    psi = np.ones(2**n)/np.sqrt(2**n)
    while (lamb <= 1. or np.isclose(lamb, 1)):
        lamb += slope
        a = np.max([1 - lamb, 0.])
        b = np.min([lamb, 1.])
        H = -a * H0 + b * H1
        U = scipy.linalg.expm(-1j*dt*H)
        psi = U @ psi

        if lamb >= k*ds:
            val, vec = np.linalg.eigh(H)
            specter.append(val)
            states_exact.append(psi)
            gs_exact.append(vec[:, 0])
            slist.append(lamb)
            k = k+1

        pbar.update(slope)
    pbar.close()

    data = {
        "state": states_exact,
        "gs": gs_exact,
        "s": slist,
        "specter": specter,
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    export_graphs(Jz, loc_fields, N_verts, N_edges, seed,
                  connect, REGULAR, d, no_local_fields, global_path)
