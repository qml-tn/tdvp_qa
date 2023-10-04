import argparse
import numpy as np
import os
import h5py
import pickle 
from tqdm import tqdm
from tenpy.tools.hdf5_io import Hdf5Saver
from tdvp_qa.generator import generate_graph, export_graphs, generate_postfix
from tdvp_qa.model import PrepareTDVP

def export_tdvp_data(data, mps, N_verts, N_edges, seed, REGULAR, d, global_path):
    if global_path is None:
        global_path = os.getcwd()
    path_mps = os.path.join(global_path,'mps/')
    if not os.path.exists(path_mps):os.makedirs(path_mps)
    path_data = os.path.join(global_path,'evolution_data/')
    if not os.path.exists(path_data):os.makedirs(path_data)
        
    postfix = generate_postfix(REGULAR,N_verts,N_edges,d,seed)    
    filename_data = os.path.join(path_data,'data'+postfix+'.pkl')
    filename_mps = os.path.join(path_mps,'mps'+postfix+'.hdf5')
    
    # Saving the final state
    with h5py.File(filename_mps,'w') as f:
        hd5saver = Hdf5Saver(f)
        mps.save_hdf5(hd5saver,f,"/")

    # Saving the evolution data
    with open(filename_data, 'wb') as f:
        pickle.dump(data, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default="data/",
                        type=str,
                        help='A full path to where the files should be stored.')
    parser.add_argument('--annealing_schedule',
                        default="linear",
                        type=str,
                        help='Annealing schedule to be used in the simulations.')
    parser.add_argument('--regular', 
                        action='store_true',
                        help='If set we generate regular graph of degree d.')
    parser.add_argument('--n_verts',
                        type=int,
                        default=16,
                        help='Number of vertices in the graph.')
    parser.add_argument('--n_edges',
                        type=int,
                        default=40,
                        help='Number of edges.')
    parser.add_argument('--dmax',
                        type=int,
                        default=8,
                        help='Maximum bond dimension to be used in the TDVP.')
    parser.add_argument('--annealing_time',
                        type=float,
                        default=10.,
                        help='Annealing time.')
    parser.add_argument('--dt',
                        type=float,
                        default=0.02,
                        help='Time step for the simulation.')
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
    
    parse_args, unknown = parser.parse_known_args()

    args_dict = parse_args.__dict__

    if len(unknown)>0:
        print('Unknown arguments: {}'.format(unknown))

    # Model generator parameters
    N_verts = args_dict['n_verts']
    N_edges = args_dict['n_edges']
    REGULAR = args_dict['regular']  #Set TRUE to generate regular graph of degree d
    d = args_dict['d'] # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    global_path  = args_dict['path']
    seed = args_dict['seed'] # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm 

    if not seed:
        seed = np.random.randint(10000)
        print(f"Using a random seed {seed}.")

    # TDVP annealing parameters
    annealing_schedule = args_dict["annealing_schedule"]
    annealing_time = args_dict["annealing_time"]
    dt = args_dict["dt"]
    Dmax = args_dict["dmax"]  

    Jz, loc_fields, connect = generate_graph(N_verts,N_edges, seed = seed, REGULAR=REGULAR, d=d, global_path=global_path) 

    assert connect != 0,  "Zero connectivity graph: it corresponds to two isolated subgraphs. The graph will not be saved and the solution will not be computed."
    
    hz = loc_fields[:,1]
    hx = np.ones(len(hz))
    eng, data, measurement = PrepareTDVP(hx,hz,Jz,annealing_schedule,Dmax,annealing_time,dt=dt)
    
    print(f"MPO CHI = {np.max(eng.model.H_MPO.chi)}")
    
    total = int(annealing_time/dt)
    t = 0
    for step in tqdm(range(total),position=0,leave=True):
        eng.run()
        t = eng.evolved_time
        data = measurement(eng, data)

    export_tdvp_data(data, eng.psi, N_verts, N_edges, seed, REGULAR, d, global_path)
    export_graphs(Jz, loc_fields, N_verts, N_edges, seed, connect, REGULAR, d, global_path)



