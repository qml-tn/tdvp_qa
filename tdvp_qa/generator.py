import os
import numpy as np
import networkx as nx

# import time

GRAPHS_PATH = 'graphs/'


def generate_graph(N_verts, N_edges=None, seed=None, REGULAR=False, d=None, no_local_fields=False, global_path=None):

    if global_path:
        path = os.path.join(global_path, GRAPHS_PATH)
        postfix = generate_postfix(
            REGULAR, N_verts, N_edges, d, seed, no_local_fields)
        filename_wm = os.path.join(path, 'weight_matrix'+postfix+'.dat')
        filename_lf = os.path.join(path, 'local_fields'+postfix+'.dat')

        if os.path.exists(filename_wm) and os.path.exists(filename_lf):
            weight_matrix = np.loadtxt(filename_wm)
            local_fields = np.loadtxt(filename_lf)
            print("Loaded generated models.")
            return weight_matrix, local_fields, True

    assert not (REGULAR and not d),  'Please specify the degree of the graph.'
    assert not (
        d and not REGULAR),  'The specified degree is not implemented as REGULAR option is not activated.'
    assert not (
        REGULAR and N_edges), 'The number of edges of a regular graph is fixed and equal to dN/2.'

    if seed != None:
        np.random.seed(seed)

    if REGULAR:
        # Number of edges is fixed by d for regular graphs
        N_edges = int(d*N_verts/2)
        graph = nx.random_regular_graph(d, N_verts, seed=seed)
        edges = np.array(sorted(graph.edges, key=lambda x: x[1]))
    else:
        graph = nx.dense_gnm_random_graph(
            N_verts, N_edges, seed=seed)  # Generate random graph
        edges = np.array(sorted(graph.edges, key=lambda x: x[1]))

    # Samples the couplings from uniform distribution between 0 and 1
    Jij = np.random.rand(N_edges)
    # Samples the local fields from uniform distribution between -0.5 and 0.5
    h_i = (np.random.rand(N_verts) - 0.5)

    weight_matrix = np.zeros((N_edges, 3))
    for i in range(N_edges):
        weight_matrix[i] = (edges.T[0, i], edges.T[1, i], Jij[i])

    local_fields = np.zeros((N_verts, 2))
    for i in range(N_verts):
        local_fields[i] = (i, h_i[i])

    if no_local_fields:
        local_fields[:, 1] = 0

    connect = nx.node_connectivity(graph)
    return weight_matrix, local_fields, connect


def export_graphs(wm, lf, N_verts, N_edges, seed, cnct, REGULAR, d, no_local_fields, global_path=None):
    if global_path is None:
        global_path = os.getcwd()
    path = os.path.join(global_path, GRAPHS_PATH)
    if not os.path.exists(path):
        os.makedirs(path)

    postfix = generate_postfix(
        REGULAR, N_verts, N_edges, d, seed, no_local_fields)
    filename_wm = os.path.join(path, 'weight_matrix'+postfix+'.dat')
    filename_lf = os.path.join(path, 'local_fields'+postfix+'.dat')

    header_wm = 'i // j // Jij --- connectivity='+str(cnct)
    header_lf = 'i // hi --- connectivity='+str(cnct)

    np.savetxt(filename_wm, wm, header=header_wm)
    np.savetxt(filename_lf, lf, header=header_lf)


def generate_postfix(REGULAR, N_verts, N_edges, d, seed, no_local_fields):
    postfix = ""
    if REGULAR:
        postfix = '_Nv_'+str(N_verts)+'_regular_d'+str(d)+'_seed_'+str(seed)
    else:
        postfix = '_Nv_'+str(N_verts)+'_N_edg_'+str(N_edges)+'_seed_'+str(seed)
    if no_local_fields:
        postfix += "_hi=0"
    return postfix
