import argparse
import numpy as np
import os
import pickle

from tdvp_qa.generator import Wishart, transverse_mpo, longitudinal_mpo, flat_sx_H0, compress_mpo
from tdvp_qa.adaptive_model_v2 import TDVP_QA_V2
from tdvp_qa.annealing_catalyst import TDVP_MULTI_MPS
from tdvp_qa.mps import initial_state_theta

from jax import config


def sum_mpos(mpo1, mpo2):
    mpo = []
    n = len(mpo1)
    for i in range(n):
        A = mpo1[i]
        B = mpo2[i]
        da = A.shape
        db = B.shape
        C = np.zeros([da[0]+db[0], da[1], da[2],
                     da[3]+db[3]], dtype=np.cdouble)
        C[:da[0], :, :, :da[3]] = A
        C[da[0]:, :, :, da[3]:] = B
        mpo.append(C)
    e = np.ones([2])
    A = np.einsum("i,i...->...", e, mpo[0])
    A = np.reshape(A, [1]+list(A.shape))
    mpo[0] = A
    A = np.einsum("i,...i->...", e, mpo[-1])
    A = np.reshape(A, list(A.shape)+[1])
    mpo[-1] = A
    return mpo


def multiply_mpos(mpo1, mpo2):
    mpo = []
    n = len(mpo1)
    for i in range(n):
        A = mpo1[i]
        B = mpo2[i]
        da = A.shape
        db = B.shape
        C = np.einsum("lijr,ajkb->laikrb", A, B)
        C = np.reshape(C, [da[0]*db[0], da[1], db[2], da[3]*db[3]])
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
    mpoxxyy = multiply_mpos(mpoxy, mpoxy)

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

    mpo = sum_mpos(mpoxy, mpoxxyy)
    compress_mpo(mpo)
    return mpo


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


def generate_tdvp_filename(p, q, global_path, annealing_schedule, Dmax, dtr,
                           dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                           rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, inith, alpha0=None,
                           seed0=None, T=None, nmps=10, reorder_mps=True, shuffle=False):
    if global_path is None:
        global_path = os.getcwd()
    path_data = os.path.join(global_path, 'factoring_catalyst/')
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    postfix = f"p_{p}_q_{q}"

    if T is not None:
        postfix += f"_{annealing_schedule}_D_{Dmax}_T_{T}_dp_{double_precision}_sl_{slope}_st_{stochastic}_sr_{seed_tdvp}_so_{slope_omega}_ri_{rand_init}"
    elif nitime > 1:
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

    postfix += f"_nmps_{nmps}"

    if reorder_mps:
        postfix += "_reord"

    if shuffle:
        postfix += "_shuffle"

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
    parser.add_argument('--alpha0',
                        type=float,
                        default=0.9,
                        help='Initial Wishart planted ensemble parameter which specifies the number-of-equations–to–number-of-variables ratio. 0.25 (default)')
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
                        default=0.2,
                        help='Imaginary time step for the simulation.')
    parser.add_argument('--nitime',
                        type=int,
                        default=10,
                        help='Number of repetitions of the same step if dti>0 or T=0.')
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
    parser.add_argument('--nmps',
                        type=int,
                        default=7,
                        help='Number of total MPS used for the evolution.')
    parser.add_argument('--recalculate',
                        action='store_true',
                        help='We restart the simulation and overwrite existing data.')
    parser.add_argument('--single_precision',
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
    parser.add_argument('--reorder_mps',
                        action='store_true',
                        help='If set we reorder the MPS with respect to the energy after each step.')
    parser.add_argument('--cyclic_path',
                        action='store_true',
                        help='If set we make a cyclic path and the final point is the same as the initial point.')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='If set we shuffle the ground state to be a random strin not a fully polarised one.')
    parser.add_argument('--inith',
                        default="flatsx",
                        type=str,
                        help='Choice of the initial Hamiltonian. sx (default), flatsx, wishart')
    parser.add_argument('--T',
                        type=float,
                        action="store",
                        help='If set we use use a Monte Carlo sampling with the temperature T of the eigenstates of the Heff instead of the real/imaginary time evolution.')

    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    # Model generator parameters
    p = args_dict['p']
    q = args_dict['q']

    nx = len(bin(p)[2:])
    ny = len(bin(q)[2:])
    n = nx + ny

    print(f"Number of qubits used: {n}")
    nmps = args_dict["nmps"]

    # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    global_path = args_dict['path']
    # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm
    seed = args_dict['seed']

    if seed is None:
        seed = np.random.randint(10000)

    inith = args_dict["inith"]
    seed0 = args_dict["seed0"]
    alpha0 = args_dict["alpha0"]
    shuffle = args_dict["shuffle"]

    if seed0 is None:
        seed0 = np.random.randint(10000)
        # print(f"Using a initial random seed {seed0}.")

    # TDVP annealing parameters
    double_precision = not args_dict["single_precision"]
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
    Tmc = args_dict["T"]
    reorder_mps = args_dict["reorder_mps"]

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

    filename = generate_tdvp_filename(p, q, global_path, annealing_schedule, Dmax, dtr,
                                      dti, slope, seed_tdvp, stochastic, double_precision, slope_omega,
                                      rand_init, rand_xy, scale_gap, auto_grad, nitime, cyclic_path, inith,
                                      alpha0, seed0, Tmc, nmps, reorder_mps)

    if recalculate:
        data = None
    else:
        data = get_simulation_data(filename)

    mpoz = factoring_mpos(p, q)

    if inith == "flatsx":
        mpox = flat_sx_H0(n)
        # we make the largest energy extensive
        # mpox[0] = mpox[0]*n
    elif inith == "sx":
        mpox = longitudinal_mpo(n, theta)
    elif inith == "wishart":
        Jx, hx, _, _ = Wishart(n, alpha0, seed0, shuffle=False)
        Jx[:, -1] = -Jx[:, -1]
        # We have to reverse the sign of Jx since the first hamiltonian comes with the - sign in front
        mpox = transverse_mpo(Jx, hx, n, rotate_to_x=True)
    else:
        mpox = longitudinal_mpo(n, theta)

    compress_mpo(mpox)

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
                                Tmc=Tmc, nmps=nmps, reorder_mps=reorder_mps)

        data = tdvpqa.evolve(data=data)
        data["mpslist"] = [[np.array(A) for A in mps.tensors]
                           for mps in tdvpqa.mpslist]
        data["p"] = p
        data["q"] = q

        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            print(f"Saved data to {filename}")

        # export_graphs(Jz, loc_fields, N_verts, N_edges, seed,
        #               connect, REGULAR, d, no_local_fields, global_path, max_cut)
        print("Simulation finished and data stored in {filename}")

    else:
        print("The simulation is already finished!")

    print(f"Final energy is: {data['energy'][-1]}")

    psilist = data["last_var_gs"]

    for psi in psilist:
        sol = [str(int(abs(ps[0, 0, 0]) < abs(ps[0, 1, 0]))) for ps in psi]
        psol = sol[:nx]
        qsol = sol[nx:]
        psol.reverse()
        qsol.reverse()
        ps = int("0b"+"".join(psol), 2)
        qs = int("0b"+"".join(qsol), 2)
        print(f"p,q = {psol}, {qsol}, {p*q==ps*qs} ")

    print(f"Solution: {bin(p)}, {bin(q)}")
    # print("Haddamard distance", (n-abs(np.sum(sol)))/2)
    print("Energy difference", np.array(data["energy"][-1])+nx+ny)
