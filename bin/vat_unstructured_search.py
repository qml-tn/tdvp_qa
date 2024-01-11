import argparse
import numpy as np
import os
import pickle

from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.mps import initial_state_theta

from jax import config


def search_mpos(n, state):
    # n is the number of spins
    # state is the designed spin configuration
    # H0
    A = np.zeros([4, 2, 2, 4], dtype=np.cdouble)
    ix = np.array([[1, 1], [1, 1]])/2.
    i2 = np.eye(2)
    A[0, :, :, 1] = -i2
    A[1, :, :, 1] = i2
    A[1, :, :, 3] = i2
    A[0, :, :, 2] = ix
    A[2, :, :, 2] = ix
    A[2, :, :, 3] = ix
    mpo0 = [A[:1, :, :, :]] + [A]*(n-2) + [A[:, :, :, -1:]]

    # H1
    A = np.zeros([4, 2, 2, 4], dtype=np.cdouble)
    A[0, :, :, 1] = i2
    A[1, :, :, 1] = i2
    A[1, :, :, 3] = i2
    mpo1 = []
    for s in state:
        Ai = A.copy()
        Ai[0, s, s, 2] = -1.0
        Ai[2, s, s, 2] = 1.0
        Ai[2, s, s, 3] = 1.0
        mpo1.append(Ai)
    mpo1[0] = mpo1[0][:1]
    mpo1[-1] = mpo1[-1][:, :, :, -1:]

    return mpo0, mpo1


def get_simulation_data(filename_path):
    data = None
    if os.path.exists(filename_path):
        with open(filename_path, 'rb') as f:
            data = pickle.load(f)
    return data


def generate_tdvp_filename(n, m, global_path, annealing_schedule, Dmax, dtr, dti, slope, seed_tdvp, stochastic, double_precision, slope_omega):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'search/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"n_{n}_m_{m}"
    postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}"
    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    return filename_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--n',
                        type=int,
                        default=16,
                        help='The number of elements in the search is N=2**n.')
    parser.add_argument('--m',
                        type=int,
                        default=16,
                        help='The vector we are looking for. Should be between 0 and 2**n-1.')
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

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    n = args_dict['n']
    m = args_dict['m']
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm

    # TDVP annealing parameters
    double_precision = args_dict["double_precision"]
    seed_tdvp = args_dict["seed_tdvp"]
    stochastic = args_dict["stochastic"]
    adaptive = args_dict["adaptive"]
    slope_omega = args_dict["slope_omega"]
    slope = args_dict["slope"]
    lamb = 0
    dtr = args_dict["dtr"]
    dti = args_dict["dti"]
    dt = dtr - 1j*dti

    Dmax = args_dict["dmax"]
    recalculate = args_dict["recalculate"]
    compute_states = args_dict["comp_state"]

    if double_precision:
        config.update("jax_enable_x64", True)

    annealing_schedule = "linear"
    if adaptive:
        annealing_schedule = "adaptive"

    theta = np.array([[np.pi/2., 0]]*n)
    tensors = initial_state_theta(n, Dmax, theta=theta)

    if m < 0 or m >= 2**n:
        raise ValueError(
            f"Stopping the simulation m={m} is too large! It should be between 0 and {2**n-1}.")

    state = [int(i) for i in ("{:0"+(f"{n}")+"b}").format(m)]
    mpo0, mpo1 = search_mpos(n, state)

    filename = generate_tdvp_filename(
        n, m, global_path, annealing_schedule, Dmax, dtr, dti, slope, seed_tdvp=seed_tdvp, stochastic=stochastic, double_precision=double_precision, slope_omega=slope_omega)

    data = get_simulation_data(filename)
    lamb = 0
    if data is not None:
        tensors = data["mps"]
        slope = data["slope"][-1]
        lamb = np.sum(data["slope"])

    if lamb < 1:
        tdvpqa = TDVP_QA_V2(mpo0, mpo1, tensors, slope, dt, lamb=lamb, max_slope=0.1, min_slope=1e-8,
                            adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega, ds=0.01)

        data = tdvpqa.evolve(data=data)
        data["mps"] = [np.array(A) for A in tdvpqa.mps.tensors]

        data["samples"] = [tdvpqa.mps.sample() for i in range(10)]
        print("Samples")
        print(data["samples"])
        print("Solution")
        print(state)

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    else:
        print("The simulation is already finished!")
