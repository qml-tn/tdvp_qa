from tdvp_qa.generator import generate_graph, export_files
import argparse
from tqdm import tqdm


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
                        default=40,
                        help='Number of edges.')
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
    
    N_verts = args_dict['n_verts']
    N_edges = args_dict['n_edges']
    REGULAR = args_dict['regular']  #Set TRUE to generate regular graph of degree d
    d = args_dict['d'] # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten
    distr = args_dict['distr'] #Sampling distribution of the local fields. It can be 'Normal' or 'Uniform'
    path  = args_dict['path']
    seed = args_dict['seed'] # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm 
    weight_matrix, loc_fields, connect = generate_graph(N_verts,N_edges, seed = seed, REGULAR=REGULAR, d=d, distr=distr) 
    
    if connect == 0:
        print("Zero connectivity graph: it corresponds to two isolated subgraphs. The graph will not be saved and the solution will not be computed.")
    else:
        export_files(weight_matrix, loc_fields, N_verts, N_edges, seed, connect, distr, REGULAR, d, path)

