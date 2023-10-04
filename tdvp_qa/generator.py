import os
import numpy as np
import networkx as nx

#import time


def generate_graph(N_verts, N_edges = None, seed = None, REGULAR = False, d = None, distr = 'Uniform'):
    
    if REGULAR and not d: 
        print('Please specify the degree of the graph.')
    if d and not REGULAR: 
        print('The specified degree is not implemented as REGULAR option is not activated.')
    if REGULAR and N_edges:
        print('The number of edges of a regular graph is fixed and equal to dN/2.')
    
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
    

def export_files(wm, lf, N_verts, seed, cnct, distr, REGULAR, d):
    curr_dir = os.getcwd()
    path = curr_dir+'/Graphs/'
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
    


################## MAIN ###################

N_instances = 10 #Number of graphs to generate
EXPORT = True   #Set TRUE to export the weighted couplings and the local fields of the graph

N_verts, N_edges = 16, 40

REGULAR = False  #Set TRUE to generate regular graph of degree d
d = None # Integer-valued degree of the regular graph. It fixes the number of edges: the value of N_edges is overwritten

distr = 'Normal' #Sampling distribution of the local fields. It can be 'Normal' or 'Uniform'

#start_time = time.time()

for n in range(N_instances):
    print('Step '+str(n+1)+'/'+str(N_instances) )
    
    seed = n # Can be integer or 'None'. If set to an integer value, it fixes the initial condition for the pseudorandom algorithm 
    
    weight_matrix, loc_fields, connect = generate_graph(N_verts,N_edges,seed = seed, REGULAR=REGULAR,d=d,distr=distr) 
    
    if connect == 0:
        print("Zero connectivity graph: it corresponds to two isolate subgraphs.")
        
    
    
    if EXPORT: #Export of two separate files for respectively the couplings matrix and the local fields. Relevant infos on the graph are also attached
        if connect == 0: print('This graph is ignored.')
        else: export_files(weight_matrix, loc_fields, N_verts, seed, connect, distr, REGULAR, d)

#print(time.time() - start_time , 'seconds')
  
    
    
    
    
    
    
    