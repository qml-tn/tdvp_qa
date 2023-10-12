import numpy as np
import os
import pandas as pd
import os
import h5py
import pickle

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import tdvp
from tenpy.tools.hdf5_io import Hdf5Saver, Hdf5Loader
from tenpy.tools.misc import setup_logging

from tdvp_qa.generator import generate_postfix

from scipy.interpolate import interp1d

np.set_printoptions(precision=5, suppress=True, linewidth=120)
setup_logging(to_stdout="ERROR")

# For the DWAVE annealing schedule
module_dir = os.path.dirname(__file__)
excel_file = os.path.join(module_dir, 'annealing_data', 'schedule.xlsx')
df = pd.read_excel(excel_file)
scale = 10**9*10**(-6)  # GHz * microsecond

# Time here is measured in milliseconds


def get_funcA(annealing_schedule):
    if annealing_schedule == "dwave":
        return interp1d(np.array(df['s']), np.array(df['A'])*scale, kind='linear')
    elif annealing_schedule == "linear":
        return lambda s: np.clip(1-s, 0, 1)
    elif annealing_schedule == "ra":
        return lambda s: 2*(0.5**2-(s-0.5)**2)
    raise Exception(
        f"Annealing schedule {annealing_schedule} not implemented.")


def get_funcB(annealing_schedule):
    if annealing_schedule == "dwave":
        return interp1d(np.array(df['s']), np.array(df['B'])*scale, kind='linear')
    elif annealing_schedule == "linear":
        return lambda s: np.clip(s, 0, 1)
    elif annealing_schedule == "ra":
        return lambda s: np.clip(s, 0, 1)
    raise Exception(
        f"Annealing schedule {annealing_schedule} not implemented.")


def get_funcC(annealing_schedule):
    if annealing_schedule == "ra":
        return lambda s: np.clip(1-s, 0, 1)
    raise Exception(
        f"Annealing schedule {annealing_schedule} not implemented.")


# Hamiltonian model
class IsingModel(CouplingMPOModel):
    """ Implementation of the Ising model which is H_1 in the annealing model without the 1/2 prefactor
    The Hamiltonian reads:

    .. math ::
        H_1 = \sum_{i<j} \mathtt{Jz}_{i,j} S^z_i S^z_j
              + \sum_i (\mathtt{hz}_i S^z_i)

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: CouplingMPOModel
        n : int (number of spins in the model)
        hz  : array
        Jz : dict
            Coupling as defined for the Hamiltonian above.
    """

    def init_sites(self, model_params):
        site = SpinHalfSite(conserve='None')
        return site

    def init_terms(self, model_params):
        hz = model_params.get('hz', [1])
        Jz = model_params.get('Jz', {})

        # H_1
        for i in range(len(hz)):
            self.add_onsite_term(
                strength=hz[i], i=i, op='Sigmaz', category=f'Sigmaz_{i}', plus_hc=False)

        if type(Jz) == dict:
            for ij in Jz.keys():
                i = np.min(ij)
                j = np.max(ij)
                self.add_coupling_term(strength=Jz[ij], i=i, j=j, op_i='Sigmaz', op_j='Sigmaz',
                                       op_string="Id", category=f'Sigmaz_{i} Sigmaz_{j}', plus_hc=False)
        elif type(Jz) == np.ndarray:
            for k in range(len(Jz)):
                ij = Jz[k, :2]
                i = int(np.min(ij))
                j = int(np.max(ij))
                self.add_coupling_term(strength=Jz[k, 2], i=i, j=j, op_i='Sigmaz', op_j='Sigmaz',
                                       op_string="Id", category=f'Sigmaz_{i} Sigmaz_{j}', plus_hc=False)
    # done


class AnnealingModel(CouplingMPOModel):
    """ Implementation of the annealing model H = A H_0 + B H H_1
    H_0 is a model with either a local x field hx which can be in a positive direction or random. 
    H_1 can be an arbitrary spin-1/2 Ising Hamiltonian with two-body couplings J_ij and local z-field hz.

    The Hamiltonian reads:

    .. math ::
        H = A H_0 + B H H_1
        H_0 = \sum_i hx_i S^x_i
        H_1 = \sum_{i<j} \mathtt{Jz}_{i,j} S^z_i S^z_j
              + \sum_i (\mathtt{hz}_i S^z_i)

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: CouplingMPOModel
        n : int (number of spins in the model)
        A, B: float
        hx, hz  : array
        Jz : dict
            Coupling as defined for the Hamiltonian above.
    """

    def init_sites(self, model_params):
        site = SpinHalfSite(conserve='None')
        return site

    def init_terms(self, model_params):
        dt = model_params.get('dt', 0.05)
        t = model_params.get('time', 0.) + dt/2.0
        annealing_time = model_params.get("annealing_time", 0.5)
        annealing_schedule = model_params.get("annealing_schedule", "dwave")
        # We use a standard linear schedule
        s = np.clip(t/annealing_time, 0, 1)

        funcA = get_funcA(annealing_schedule)
        funcB = get_funcB(annealing_schedule)
        A = funcA(s)
        B = funcB(s)
        hx = model_params.get('hx', [1])
        hz = model_params.get('hz', [1])
        Jz = model_params.get('Jz', {})

        # H_0
        for i in range(len(hx)):
            self.add_onsite_term(
                strength=-A*hx[i]/2., i=i, op='Sigmax', category=f'Sigmax_{i}', plus_hc=False)

        # H_1
        for i in range(len(hz)):
            self.add_onsite_term(
                strength=B*hz[i]/2., i=i, op='Sigmaz', category=f'Sigmaz_{i}', plus_hc=False)

        if type(Jz) == dict:
            for ij in Jz.keys():
                i = np.min(ij)
                j = np.max(ij)
                self.add_coupling_term(strength=B*Jz[ij]/2., i=i, j=j, op_i='Sigmaz', op_j='Sigmaz',
                                       op_string="Id", category=f'Sigmaz_{i} Sigmaz_{j}', plus_hc=False)
        elif type(Jz) == np.ndarray:
            for k in range(len(Jz)):
                ij = Jz[k, :2]
                i = int(np.min(ij))
                j = int(np.max(ij))
                self.add_coupling_term(strength=B*Jz[k, 2]/2., i=i, j=j, op_i='Sigmaz', op_j='Sigmaz',
                                       op_string="Id", category=f'Sigmaz_{i} Sigmaz_{j}', plus_hc=False)
    # done


# Hamiltonian model
class ReverseAnnealingModel(CouplingMPOModel):
    """ Implementation of the annealing model H = A H_0 + B H_1 + C H_2
    H_0 is a model with either a local x field hx which can be in a positive direction or random. 
    H_1 can be an arbitrary spin-1/2 Ising Hamiltonian with two-body couplings J_ij and local z-field hz.
    H_2 is the initial model along z direction that should be close to the solution of H_1

    The Hamiltonian reads:

    .. math ::
        H = A H_0 + B H_1 + C H_2
        H_0 = \sum_i hx_i S^x_i
        H_1 = \sum_{i<j} \mathtt{Jz}_{i,j} S^z_i S^z_j
              + \sum_i (\mathtt{hz}_i S^z_i)
        H_2 = \sum_i (\mathtt{hz0}_i S^z_i)

    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: CouplingMPOModel
        n : int (number of spins in the model)
        A, B, C, C0: float
        hx, hz, hz0  : array
        Jz : dict
            Coupling as defined for the Hamiltonian above.
    """

    def init_sites(self, model_params):
        site = SpinHalfSite(conserve='None')
        return site

    def init_terms(self, model_params):
        dt = model_params.get('dt', 0.05)
        t = model_params.get('time', 0.) + dt/2.0
        annealing_time = model_params.get("annealing_time", 0.5)
        # We use a standard linear schedule
        s = np.clip(t/annealing_time, 0, 1)

        A0 = model_params.get('A0', 1.)
        B0 = model_params.get('B0', 1.)
        C0 = model_params.get('C0', 1.)

        annealing_schedule = "ra"
        funcA = get_funcA(annealing_schedule)
        funcB = get_funcB(annealing_schedule)
        funcC = get_funcC(annealing_schedule)

        A = funcA(s) * A0
        B = funcB(s) * B0
        C = funcC(s) * C0

        hx = model_params.get('hx', [1])
        hz = model_params.get('hz', [1])
        hz0 = model_params.get('hz0', [1])
        Jz = model_params.get('Jz', {})

        # H_0
        for i in range(len(hx)):
            self.add_onsite_term(
                strength=-A*hx[i]/2., i=i, op='Sigmax', category=f'Sigmax_{i}', plus_hc=False)

        # H_1
        for i in range(len(hz)):
            self.add_onsite_term(
                strength=B*hz[i]/2., i=i, op='Sigmaz', category=f'Sigmaz_{i}', plus_hc=False)

        for ij in Jz.keys():
            i = np.min(ij)
            j = np.max(ij)
            self.add_coupling_term(strength=B*Jz[ij]/2., i=i, j=j, op_i='Sigmaz', op_j='Sigmaz',
                                   op_string="Id", category=f'Sigmaz_{ij[0]} Sigmaz_{ij[1]}', plus_hc=False)

        # H_2
        for i in range(len(hz0)):
            self.add_onsite_term(
                strength=-C*hz0[i]/2., i=i, op='Sigmaz', category=f'Sigmaz0_{i}', plus_hc=False)

    # done


# Construction of the initial state
# Helper functions
def bond_dimensions(n, d, Dmax):
    '''
    List of bond dimensions

    Parameters: 
      - n     : length of the system
      - d     : local Hilbert space size
      - Dmax  : maximum bond dimension
    '''
    dims = []
    nhalf = int((n+1)//2)
    for i in range(nhalf):
        dims.append(int(np.min([d**i, Dmax])))
    middle = []
    if np.mod(n, 2) == 0:
        middle = [int(np.min([d**nhalf, Dmax]))]
    return dims + middle + dims[::-1]


def initial_state(hx, Dmax):
    '''
    List of MPS matrices for a particular initial product state

    Parameters:
      - hx     : local magnetic fields of H_0
      - Dmax  : maximum bond dimension
    '''
    n = len(hx)
    d = 2
    Bs = []
    SVs = []
    dims = bond_dimensions(n, d, Dmax)

    for i in range(n):
        B = np.zeros([d, dims[i], dims[i+1]])
        v = np.array([1., np.sign(hx[i])])/np.sqrt(2.)
        B[:, 0, 0] = v
        Bs.append(B)

    for D in dims:
        s = np.zeros(D)
        s[0] = 1
        SVs.append(s)
    return Bs, SVs, dims


def initial_state_RA(hz0, Dmax):
    '''
    List of MPS matrices for a particular initial product state

    Parameters:
      - hz0     : local transverse magnetic fields of H_2 should have values -1 or 1
      - Dmax  : maximum bond dimension
    '''
    n = len(hz0)
    d = 2
    Bs = []
    SVs = []
    dims = bond_dimensions(n, d, Dmax)

    for i in range(n):
        B = np.zeros([d, dims[i], dims[i+1]])
        s = hz0[i]
        v = np.array([(1.+s)/2., (1.-s)/2.])
        B[:, 0, 0] = v
        Bs.append(B)

    for D in dims:
        s = np.zeros(D)
        s[0] = 1
        SVs.append(s)
    return Bs, SVs, dims

# Preparing the annealing engine


def PrepareTDVP(hx, hz, Jz, N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, annealing_time, dt):
    '''
      Main function that prepares the initial product state with bond dimension D for the annealing process with tenpy
      and creates the Hamiltonian associated with the couplings J and onsite potential h

      Parameters:
      - hx       : onsite magnetic field along x
      - hz       : onsite magnetic field along z
      - Jz       : couplings between sites either a dict or an np.array
      - annealing_schedule: annealing schedule
      - Dmax     : maximum bond dimension
      - annealing_time     : annealing time
    '''

    # Annealing model
    nx = len(hx)
    nz = len(hz)
    L = np.max([nx, nz])

    model_params = {
        'Jz': Jz,
        'hz': hz,
        'hx': hx,
        'annealing_time': annealing_time,
        'bc_MPS': 'finite',
        'L': L,
        'annealing_schedule': annealing_schedule,
        'dt': dt,
    }

    M = AnnealingModel(model_params)

    data, psi = import_tdvp_data(N_verts, N_edges, seed, REGULAR, d, no_local_fields,
                                 global_path, annealing_schedule, Dmax, annealing_time, dt)
    if psi is None:
        # Model sites
        sites = M.lat.mps_sites()

        # initial_state
        Bflat, SVs, _ = initial_state(hx, Dmax)
        psi = MPS.from_Bflat(sites, Bflat, SVs)
        start_time = 0
    else:
        start_time = data['t'][-1]

    # TDVP
    tdvp_params = {
        'N_steps': 1,
        'dt': dt,
        'preserve_norm': True,
        'start_time': start_time,
        # 'trunc_params': {'chi_max': 16}
    }

    eng = tdvp.TimeDependentSingleSiteTDVP(psi, M, tdvp_params)

    def measurement(eng, data, dmrg_energy=False):
        keys = ['t', 'entropy', 'energy', 'dmrg_energy', 'xmean', 'zmean']

        if data is None:
            data = dict([(k, []) for k in keys])

        if dmrg_energy:
            data['dmrg_energy'].append(eng.model.H_MPO.expectation_value(psi))
        else:
            data['t'].append(eng.evolved_time)
            data['entropy'].append(psi.entanglement_entropy(bonds=[L//2])[0])
            data['energy'].append(eng.model.H_MPO.expectation_value(psi))
            data['xmean'].append(
                np.mean(np.abs(psi.expectation_value('Sigmax'))))
            data['zmean'].append(
                np.mean(np.abs(psi.expectation_value('Sigmaz'))))
        return data

    if data is None:
        data = measurement(eng, None)

    return eng, data, measurement

# Preparing the annealing engine


def PrepareReverseTDVP(hx, hz, Jz, hz0, Dmax, annealing_time, dt=0.1, A0=1, B0=1, C0=1.):
    '''
      Main function that prepares the initial product state with bond dimension D for the reversed annealing process with tenpy
      and creates the Hamiltonian associated with the couplings J and onsite potential h

      Parameters:
      - hx       : onsite magnetic field along x
      - hz       : onsite magnetic field along z
      - Jz       : couplings between sites
      - hz0      : initial state magnetization in the z direction
      - Dmax     : maximum bond dimension
      - annealing_time     : annealing time
    '''

    # Annealing model
    nx = len(hx)
    nz = len(hz)
    L = np.max([nx, nz])

    model_params = {
        'Jz': Jz,
        'hz': hz,
        'hx': hx,
        'hz0': hz0,
        'A0': A0,
        'B0': B0,
        'C0': C0,
        'annealing_time': annealing_time,
        'bc_MPS': 'finite',
        'L': L,
    }

    M = ReverseAnnealingModel(model_params)

    # Model sites
    sites = M.lat.mps_sites()

    # initial_state
    Bflat, SVs, _ = initial_state_RA(hz0, Dmax)
    psi = MPS.from_Bflat(sites, Bflat, SVs)

    # TDVP
    tdvp_params = {
        'N_steps': 1,
        'dt': dt,
        'preserve_norm': True,
        'start_time': 0,
    }

    eng = tdvp.TimeDependentSingleSiteTDVP(psi, M, tdvp_params)

    def measurement(eng, data):
        keys = ['t', 'entropy', 'energy', 'xmean', 'zmean']
        if data is None:
            data = dict([(k, []) for k in keys])
        data['t'].append(eng.evolved_time)
        data['entropy'].append(psi.entanglement_entropy(bonds=[L//2])[0])
        data['energy'].append(eng.model.H_MPO.expectation_value(psi))
        data['xmean'].append(np.mean(np.abs(psi.expectation_value('Sigmax'))))
        data['zmean'].append(np.mean(np.abs(psi.expectation_value('Sigmaz'))))
        return data

    data = measurement(eng, None)

    return eng, data, measurement


def export_tdvp_data(data, mps, N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, annealing_time, dt):
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
    postfix += f"_{annealing_schedule}_D_{Dmax}_t_{annealing_time}_dt_{dt}"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    filename_mps = os.path.join(path_mps, 'mps'+postfix+'.hdf5')

    # Saving the final state
    with h5py.File(filename_mps, 'w') as f:
        hd5saver = Hdf5Saver(f)
        mps.save_hdf5(hd5saver, f, "/")

    # Saving the evolution data
    with open(filename_data, 'wb') as f:
        pickle.dump(data, f)


def import_tdvp_data(N_verts, N_edges, seed, REGULAR, d, no_local_fields, global_path, annealing_schedule, Dmax, annealing_time, dt):
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
    postfix += f"_{annealing_schedule}_D_{Dmax}_t_{annealing_time}_dt_{dt}"

    filename_data = os.path.join(path_data, 'data'+postfix+'.pkl')
    filename_mps = os.path.join(path_mps, 'mps'+postfix+'.hdf5')

    data = None
    mps = None
    if os.path.exists(filename_data) and os.path.exists(filename_mps):
        # Opening the data
        try:
            with h5py.File(filename_mps, 'r') as f:
                hd5loader = Hdf5Loader(f)
                mps = MPS.from_hdf5(hd5loader, f, "/")
            with open(filename_data, 'rb') as f:
                data = pickle.load(f)
        except:
            print("Loading failed continue with new data!")
    return data, mps
