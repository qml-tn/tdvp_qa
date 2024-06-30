import argparse
import numpy as np
import os
import pickle

from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.mps import initial_state_theta
from tdvp_qa.generator import longitudinal_mpo

from jax import config

def sum_mpos(mpo1,mpo2):
    mpo = []
    n = len(mpo1)
    for i in range(n):
        A = mpo1[i]
        B = mpo2[i]
        da = A.shape
        db = B.shape
        C = np.zeros([da[0]+db[0],da[1],da[2],da[3]+db[3]], dtype=np.cdouble)
        C[:da[0],:,:,:da[3]] = A
        C[da[0]:,:,:,da[3]:] = B
        mpo.append(C)
    e = np.ones([2])    
    A = np.einsum("i,i...->...",e,mpo[0])
    A = np.reshape(A,[1]+list(A.shape))
    mpo[0] = A
    A = np.einsum("i,...i->...",e,mpo[-1])
    A = np.reshape(A,list(A.shape)+[1])
    mpo[-1] = A
    return mpo

def multiply_mpos(mpo1,mpo2):
    mpo= []
    n = len(mpo1)
    for i in range(n):
        A = mpo1[i]
        B = mpo2[i]
        da = A.shape
        db = B.shape
        C = np.einsum("lijr,ajkb->laikrb",A,B)
        C = np.reshape(C,[da[0]*db[0],da[1],db[2],da[3]*db[3]])
        mpo.append(C)
    return mpo

def factoring_mpos(p, q):
    n = p*q
    nx = len(bin(p)[2:])
    ny = len(bin(q)[2:])

    
    A = np.zeros([2, 2, 2, 2], dtype=np.cdouble)
    F = np.zeros([2, 2, 2, 1], dtype=np.cdouble)
    sp = np.array([[0, 0], [0, 1]])
    i2 = np.eye(2)

    A[0, :, :, 0] = 2*i2
    A[1, :, :, 1] = i2
    A[0, :, :, 1] = sp
    
    F[0, :, :, 0] = sp
    F[1, :, :, 0] = i2

    mpoxy = [A[:1]] + [A]*(nx-2) + [F[:, :, :, :], A[:1]] + [A]*(ny-2) + [F]
    mpoxxyy = multiply_mpos(mpoxy,mpoxy)

    # mpoxy[0] = (-2*n*mpoxy[0])*(nx+ny)/n**2 # We simplify this expression below
    
    # mpoxy[0] = (-2*mpoxy[0])*(nx+ny)/n
    # mpoxxyy[0] = (mpoxxyy[0])*(nx+ny)/n**2

    mpoxy[0] = (-2*mpoxy[0])
    m = nx+ny
    c1 = ((nx+ny)/n)**(1./m)
    c2 = ((nx+ny)/n**2)**(1./m)
    for i in range(m):
        mpoxy[i] = c1*mpoxy[i]        
        mpoxxyy[i] = c2*mpoxxyy[i]        

    return sum_mpos(mpoxy,mpoxxyy)


def get_simulation_data(filename_path):
    data = None
    if os.path.exists(filename_path):
        with open(filename_path, 'rb') as f:
            data = pickle.load(f)
    return data


def generate_tdvp_filename(p,q, global_path, annealing_schedule, Dmax, dtr, dti, slope, seed_tdvp, stochastic, double_precision, slope_omega, scale_gap, nitime):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'factoring/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"p_{p}_q_{q}"
    postfix += f"_{annealing_schedule}_D_{Dmax}_dt_{dtr}_{dti}_{nitime}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}"
    if scale_gap:
        postfix += "_sgap"
    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    return filename_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--p',
                        type=int,
                        default=5,
                        help='The First factor of the composit number')
    parser.add_argument('--q',
                        type=int,
                        default=13,
                        help='The second factor of the composit number.')
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
    parser.add_argument('--schedule',
                        default="linear",
                        type=str,
                        help='Annealing schedule: linear (default), adaptive, quadratic.')
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
    parser.add_argument('--nitime',
                        type=int,
                        default=1,
                        help='Number of repetitions of the same step if dti>0.')
    parser.add_argument('--scale_gap',
                        action='store_true',
                        help='If set we use gap scaling. This can be used with or without adaptive step adjustment.')

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    p = args_dict['p']
    q = args_dict['q']

    nx = len(bin(p)[2:])
    ny = len(bin(q)[2:])
    n = nx + ny

    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm

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

    if dti > 0:
        nitime = args_dict["nitime"]
    else:
        nitime = 0

    Dmax = args_dict["dmax"]
    recalculate = args_dict["recalculate"]
    compute_states = args_dict["comp_state"]

    if double_precision:
        config.update("jax_enable_x64", True)

    annealing_schedule = args_dict["schedule"]
    if annealing_schedule == "adaptive":
        adaptive = True
    elif adaptive:
        annealing_schedule = "adaptive"

    theta = np.array([[np.pi/2., 0]]*n)
    tensors = initial_state_theta(n, Dmax, theta=theta)

    mpo0 = longitudinal_mpo(n, theta)
    mpo1 = factoring_mpos(p,q)

    tensors = initial_state_theta(n, Dmax, theta=theta)


    filename = generate_tdvp_filename(
        p, q, global_path, annealing_schedule, Dmax, dtr, dti, slope, seed_tdvp=seed_tdvp, stochastic=stochastic, double_precision=double_precision, slope_omega=slope_omega, scale_gap=scale_gap, nitime=nitime)

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
        tdvpqa = TDVP_QA_V2(mpo0, mpo1, tensors, slope, dt, lamb=lamb, max_slope=0.1, min_slope=1e-10,
                            adaptive=adaptive, compute_states=compute_states, key=seed_tdvp, slope_omega=slope_omega, ds=0.01, scale_gap=scale_gap, nitime=nitime)

        data = tdvpqa.evolve(data=data)
        data["mps"] = [np.array(A) for A in tdvpqa.mps.tensors]

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    else:
        print("The simulation is already finished!")

    print("Energy difference",nx+ny+data["energy"][-1])