import os
import numpy as np
import networkx as nx

# import time

GRAPHS_PATH = 'graphs/'


def generate_graph(N_verts, N_edges=None, seed=None, REGULAR=False, d=None, no_local_fields=False, global_path=None, recalculate=False):

    if global_path:
        path = os.path.join(global_path, GRAPHS_PATH)
        postfix = generate_postfix(
            REGULAR, N_verts, N_edges, d, seed, no_local_fields)
        filename_wm = os.path.join(path, 'weight_matrix'+postfix+'.dat')
        filename_lf = os.path.join(path, 'local_fields'+postfix+'.dat')

        if not recalculate and os.path.exists(filename_wm) and os.path.exists(filename_lf):
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


def longitudinal_mpo(n):
    A = np.zeros([2, 2, 2, 2])
    sx = np.array([[0, 1], [1, 0]])
    i2 = np.eye(2)
    A[0, :, :, 0] = i2
    A[1, :, :, 1] = i2
    A[0, :, :, 1] = sx
    mpo = [A[:1]] + [A]*(n-2)+[A[:, :, :, -1:]]
    return mpo


def transverse_mpo(Jz, hz, n):
    d = 2
    sz = np.array([[1, 0], [0, -1]])
    i2 = np.eye(d)
    A = np.zeros([n+1, d, d, n+1])
    A[0, :, :, 0] = i2
    A[-1, :, :, -1] = i2
    mpo = [A.copy() for i in range(n)]
    # Local fields
    for i in range(n):
        mpo[i][0, :, :, -1] += sz*hz[i]
        if i > 0:
            mpo[i][i, :, :, -1] += sz
        for j in range(1, i):
            mpo[j][i, :, :, i] += i2
    # Interaction
    for k in range(len(Jz)):
        i = int(Jz[k, 0])-1
        j = int(Jz[k, 1])-1
        Jij = Jz[k, 2]
        mpo[i][0, :, :, j] += sz*Jij

    # Taking only the output state of the last mpo tensor
    mpo[0] = mpo[0][:1]
    mpo[-1] = mpo[-1][:, :, :, -1:]

    # Bring the mpo in the left canonical form
    Al = mpo[0]
    for i in range(n-1):
        Dl, _, _, Dr = Al.shape
        Al = np.reshape(Al, [-1, Dr])
        q, r = np.linalg.qr(Al, mode="reduced")
        Dr = q.shape[-1]
        mpo[i] = np.reshape(q, [Dl, d, d, Dr])
        Al = np.einsum("ij,jklm->iklm", r, mpo[i+1])
        mpo[i+1] = Al

    # Compress MPO
    Ar = mpo[-1]
    eps = 1e-12
    for i in range(n-1, 0, -1):
        Dl, _, _, Dr = Ar.shape
        Ar = np.reshape(Ar, [Dl, -1])
        u, s, v = np.linalg.svd(Ar, full_matrices=False)
        smax = np.max(s)
        snorm = s/smax
        inds = snorm > eps
        s = s[inds]
        v = v[inds]
        u = u[:, inds]
        Dl = len(s)
        mpo[i] = np.reshape(v, [Dl, d, d, Dr])
        Ar = np.einsum("ijkl,lm,m->ijkm", mpo[i-1], u, s)
        mpo[i-1] = Ar

    return mpo
