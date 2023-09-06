import numpy as np
import argparse
import os
import pandas as pd

from tqdm import tqdm
from scipy.interpolate import interp1d
np.set_printoptions(precision=5, suppress=True, linewidth=120)

import tenpy
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS

from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import asConfig

tenpy.tools.misc.setup_logging(to_stdout="ERROR")

module_dir = os.path.dirname(__file__)

excel_file = os.path.join(module_dir, 'annealing_data','schedule.xlsx')  # Replace with the path to your Excel file
df = pd.read_excel(excel_file)

scale = 10**9*10**(-6) # GHz * millisecond

# Time here is measured in milliseconds
funcA = interp1d(np.array(df['s']), np.array(df['A'])*scale, kind='linear')
funcB = interp1d(np.array(df['s']), np.array(df['B'])*scale, kind='linear')

# Hamiltonian model
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
    site = SpinHalfSite(conserve = 'None')
    return site

  def init_terms(self, model_params):
    t = model_params.get('time',0.)
    tmax = model_params.get("tmax",1.)
    s = np.clip(t/tmax,0,1) # We use a standard linear schedule
    A = funcA(s)
    B = funcB(s)
    hx = model_params.get('hx',[1])
    hz = model_params.get('hz',[1])
    Jz = model_params.get('Jz',{})

    # H_0
    for i in range(len(hx)):
      self.add_onsite_term(strength=-A*hx[i],i=i,op='Sx',category=f'Sx_{i}',plus_hc=False)

    # H_1
    for i in range(len(hz)):
      self.add_onsite_term(strength=B*hz[i],i=i,op='Sz',category=f'Sz_{i}',plus_hc=False)

    for ij in Jz:
      self.add_coupling_term(strength=B*Jz[ij],i=ij[0],j=ij[1],op1='Sz',op2='Sz',op_string="Id",category=f'Sz_{ij[0]} Sz_{ij[1]}', plus_hc=False)
  # done



  # Construction of the initial state
# Helper functions
def bond_dimensions(n,d,Dmax):
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
      dims.append(int(np.min([d**i,Dmax])))
  middle = []
  if np.mod(n,2)==0:
      middle = [int(np.min([d**nhalf,Dmax]))]
  return dims + middle + dims[::-1]

def initial_state(hx,Dmax):
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
  dims = bond_dimensions(n,d,Dmax)

  for i in range(n):
    B = np.zeros([d,dims[i],dims[i+1]])
    v = np.array([1.,np.sign(hx[i])])/np.sqrt(2.)
    B[:,0,0] = v
    Bs.append(B)
          
  for D in dims:
    s = np.zeros(D)
    s[0] = 1
    SVs.append(s)
  return Bs, SVs, dims

# Main function
def PrepareTDVP(hx,hz,Jz,Dmax,tmax,dt=0.1):
  '''
    Main function that prepares the initial product state with bond dimension D and initial for simulation with tenpy
    and creates the Hamiltonian associated with the couplings J and onsite potential h
    
    Parameters:
    - hx       : onsite magnetic field along x
    - hz       : onsite magnetic field along z
    - Jz       : couplings between sites
    - Dmax     : maximum bond dimension
    - tmax     : annealing time
  '''
  
  # Annealing model
  nx = len(hx)
  nz = len(hz)
  L = np.max([nx,nz])

  model_params= {
      'Jz': Jz,
      'hz': hz,
      'hx': hx,
      'tmax': tmax,
      'bc_MPS': 'finite',
      'L': L,
  }

  M = AnnealingModel(model_params)

  # Model sites
  sites = M.lat.mps_sites()

  # initial_state
  Bflat,SVs,dims = initial_state(hx,Dmax)
  psi = MPS.from_Bflat(sites,Bflat,SVs)

  # TDVP
  tdvp_params = {
    'N_steps': 1,
    'dt': dt,
    'preserve_norm': True,
    'start_time': 0,
    # 'trunc_params': {'chi_max': 16}
  }

  eng = tdvp.TimeDependentSingleSiteTDVP(psi, M, tdvp_params)
  
  def measurement(eng, data):
    keys = ['t', 'entropy','energy','sx']
    if data is None:
      data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['entropy'].append(psi.entanglement_entropy(bonds=[L//2])[0])
    data['energy'].append(eng.model.H_MPO.expectation_value(psi))
    data['sx'].append(psi.expectation_value('Sx'))
    return data

  data = measurement(eng, None)
  return eng, data, measurement
  