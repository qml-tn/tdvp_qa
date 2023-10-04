import os
import numpy as np
import networkx as nx

#import time


def generate_graph(N_verts, N_edges = None, seed = None, REGULAR = False, d = None, distr = 'Uniform'):
    
    assert not (REGULAR and not d),  'Please specify the degree of the graph.'
    assert not (d and not REGULAR),  'The specified degree is not implemented as REGULAR option is not activated.'
    assert not (REGULAR and N_edges), 'The number of edges of a regular graph is fixed and equal to dN/2.'
    
    if seed != None:
        np.random.seed(seed) 
        
    if REGULAR : 
        N_edges = int(d*N_verts/2) #Number of edges is fixed by d for regular graphs
        graph = nx.random_regular_graph(d,N_verts,seed = seed)
        edges = np.array(sorted(graph.edges, key=lambda x: x[1]))
    else:
        graph = nx.dense_gnm_random_graph(N_verts, N_edges, seed = seed) #Generate random graph
        edges = np.array(sorted(graph.edges, key=lambda x: x[1]))
    
    Jij = np.random.rand(N_edges) #Samples the couplings from uniform distribution between 0 and 1

    if distr == 'Uniform':
        h_i = (np.random.rand(N_verts) - 0.5) #Samples the local fields from uniform distribution between -0.5 and 0.5
    
    elif distr == 'Normal':
        h_i = np.random.normal(loc = 0, scale = 1., size = N_verts)  #Samples the local fields from normal distribution mu=0 and sigma=1

    else: print("Distribution not available")
    
    weight_matrix = np.zeros((N_edges, 3))
    for i in range(N_edges): weight_matrix[i] = (edges.T[0,i], edges.T[1,i], Jij[i])  
    
    local_fields = np.zeros((N_verts, 2))
    for i in range(N_verts): local_fields[i] = (i, h_i[i])  
    
    connect = nx.node_connectivity(graph)
    return weight_matrix, local_fields, connect
    
def export_files(wm, lf, N_verts, N_edges, seed, cnct, distr, REGULAR, d, path=None):
    if path is None:
        path = os.getcwd()
    path = os.path.join(path,'Graphs/')
    if not os.path.exists(path):os.makedirs(path)
        
    if REGULAR:
        filename_wm = path+'weight_matrix_Nv_'+str(N_verts)+'_regular_d'+str(d)+'_seed_'+str(seed)+'.dat'
        filename_lf = path+'local_fields_Nv_'+str(N_verts)+'_regular_d'+str(d)+'_seed_'+str(seed)+'.dat'
    else:
        filename_wm = path+'weight_matrix_Nv_'+str(N_verts)+'_N_edg_'+str(N_edges)+'_seed_'+str(seed)+'.dat'
        filename_lf = path+'local_fields_Nv_'+str(N_verts)+'_N_edg_'+str(N_edges)+'_seed_'+str(seed)+'.dat'

    header_wm = 'i // j // Jij --- connectivity='+str(cnct)
    header_lf = 'i // hi --- connectivity='+str(cnct)+' hi_distribution='+distr
    
    np.savetxt(filename_wm, wm, header = header_wm)
    np.savetxt(filename_lf, lf, header = header_lf)
    
  
    
    
    
    
    
    
    