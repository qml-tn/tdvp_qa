import os
import numpy as np
import networkx as nx
from scipy.stats import ortho_group


# import time

GRAPHS_PATH = 'graphs/'


def is_symmetric(matrix):
    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-08)
    return is_symmetric


def generate_graph(N_verts, N_edges=None, seed=None, REGULAR=False, d=None, no_local_fields=False, global_path=None, recalculate=False, max_cut=False):

    if global_path:
        path = os.path.join(global_path, GRAPHS_PATH)
        postfix = generate_postfix(
            REGULAR, N_verts, N_edges, d, seed, no_local_fields, max_cut=max_cut)
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

    if max_cut:
        Jij[:] = 1.0

    # Samples the local fields from uniform distribution between -0.5 and 0.5
    h_i = (np.random.rand(N_verts) - 0.5)

    weight_matrix = np.zeros((N_edges, 3))
    for i in range(N_edges):
        weight_matrix[i] = (edges.T[0, i], edges.T[1, i], Jij[i])

    local_fields = np.zeros((N_verts, 2))
    for i in range(N_verts):
        local_fields[i] = (i, h_i[i])

    if no_local_fields or max_cut:
        local_fields[:, 1] = 0

    connect = nx.node_connectivity(graph)
    return weight_matrix, local_fields, connect


def export_graphs(wm, lf, N_verts, N_edges, seed, cnct, REGULAR, d, no_local_fields, global_path=None, max_cut=False):
    if global_path is None:
        global_path = os.getcwd()
    path = os.path.join(global_path, GRAPHS_PATH)
    if not os.path.exists(path):
        os.makedirs(path)

    postfix = generate_postfix(
        REGULAR, N_verts, N_edges, d, seed, no_local_fields, max_cut)
    filename_wm = os.path.join(path, 'weight_matrix'+postfix+'.dat')
    filename_lf = os.path.join(path, 'local_fields'+postfix+'.dat')

    header_wm = 'i // j // Jij --- connectivity='+str(cnct)
    header_lf = 'i // hi --- connectivity='+str(cnct)

    np.savetxt(filename_wm, wm, header=header_wm)
    np.savetxt(filename_lf, lf, header=header_lf)


def generate_postfix(REGULAR, N_verts, N_edges, d, seed, no_local_fields, max_cut):
    postfix = ""
    if REGULAR:
        postfix = '_Nv_'+str(N_verts)+'_regular_d'+str(d)+'_seed_'+str(seed)
    else:
        postfix = '_Nv_'+str(N_verts)+'_N_edg_'+str(N_edges)+'_seed_'+str(seed)
    if no_local_fields:
        postfix += "_hi=0"
    if max_cut:
        postfix += "_mc"
    return postfix


def longitudinal_mpo(n, hx=None):
    A = np.zeros([2, 2, 2, 2])
    sx = np.array([[0, 1], [1, 0]])
    i2 = np.eye(2)
    A[0, :, :, 0] = i2
    A[1, :, :, 1] = i2
    A[0, :, :, 1] = sx
    if hx is None:
        mpo = [A[:1]] + [A]*(n-2)+[A[:, :, :, -1:]]
    else:
        Ai = A.copy()
        Ai[0, :, :, 1] *= hx[0]
        mpo = [Ai[:1]]
        for i in range(1, n-1):
            Ai = A.copy()
            Ai[0, :, :, 1] *= hx[i]
            mpo.append(Ai)
        Ai = A.copy()
        Ai[0, :, :, 1] *= hx[n-1]
        mpo.append(Ai[:, :, :, -1:])
    return mpo


def h0_theta(theta=None):
    sx = np.array([[0, 1], [1, 0]])
    if theta is None:
        return sx
    th = theta[0]
    fi = theta[1]
    return np.array([[np.cos(th), np.sin(th)*np.exp(1j*fi)], [np.sin(th)*np.exp(-1j*fi), -np.cos(th)]])


def longitudinal_mpo(n, theta, dtype=np.cdouble, g=1.0):
    A = np.zeros([2, 2, 2, 2], dtype=dtype)
    i2 = np.eye(2)
    A[0, :, :, 0] = i2
    A[1, :, :, 1] = i2
    Ai = A.copy()
    Ai[0, :, :, 1] = h0_theta(theta[0])*g
    mpo = [Ai[:1]]
    for i in range(1, n-1):
        Ai = A.copy()
        Ai[0, :, :, 1] = h0_theta(theta[i])*g
        mpo.append(Ai)
    Ai = A.copy()
    Ai[0, :, :, 1] = h0_theta(theta[n-1])*g
    mpo.append(Ai[:, :, :, -1:])
    return mpo


def transverse_mpo(Jz, hz, n, rotate_to_x=False, dtype=np.cdouble):
    d = 2
    sz = np.array([[1, 0], [0, -1]], dtype=dtype)
    if rotate_to_x:
        # we rotate the MPO to the x direction
        sz = np.array([[0, 1], [1, 0]], dtype=dtype)

    i2 = np.eye(d, dtype=dtype)
    A = np.zeros([n+1, d, d, n+1], dtype=dtype)
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
        # i = int(Jz[k, 0])
        # j = int(Jz[k, 1])
        i = int(np.min(Jz[k, :2]))
        j = int(np.max(Jz[k, :2]))
        Jij = Jz[k, 2]
        mpo[i][0, :, :, j] += sz*Jij

    # Taking only the output state of the last mpo tensor
    mpo[0] = mpo[0][:1]
    mpo[-1] = mpo[-1][:, :, :, -1:]

    # Bring the mpo in the left canonical form
    Al = mpo[0]
    norms = []
    for i in range(n-1):
        Dl, _, _, Dr = Al.shape
        Al = np.reshape(Al, [-1, Dr])
        q, r = np.linalg.qr(Al, mode="reduced")
        nrm = np.linalg.norm(r)
        # We remove the norms and store them in order to avoid overflow
        r = r/nrm
        norms.append(nrm)
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
        # We bring back the norms to restore the norm of the entire MPO
        mpo[i] = np.reshape(v, [Dl, d, d, Dr])*norms[i-1]
        Ar = np.einsum("ijkl,lm,m->ijkm", mpo[i-1], u, s)
        mpo[i-1] = Ar
    return mpo


def ising_with_field_mpo(Jz, hz, hx, n, dtype=np.cdouble):
    d = 2
    sz = np.array([[1, 0], [0, -1]], dtype=dtype)
    sx = np.array([[0, 1], [1, 0]], dtype=dtype)

    i2 = np.eye(d, dtype=dtype)
    A = np.zeros([n+1, d, d, n+1], dtype=dtype)
    A[0, :, :, 0] = i2
    A[-1, :, :, -1] = i2
    mpo = [A.copy() for i in range(n)]
    # Local fields
    for i in range(n):
        mpo[i][0, :, :, -1] += sz*hz[i]+sx*hx[i]
        if i > 0:
            mpo[i][i, :, :, -1] += sz
        for j in range(1, i):
            mpo[j][i, :, :, i] += i2
    # Interaction
    for k in range(len(Jz)):
        # i = int(Jz[k, 0])
        # j = int(Jz[k, 1])
        i = int(np.min(Jz[k, :2]))
        j = int(np.max(Jz[k, :2]))
        Jij = Jz[k, 2]
        mpo[i][0, :, :, j] += sz*Jij

    # Taking only the output state of the last mpo tensor
    mpo[0] = mpo[0][:1]
    mpo[-1] = mpo[-1][:, :, :, -1:]

    # Bring the mpo in the left canonical form
    Al = mpo[0]
    norms = []
    for i in range(n-1):
        Dl, _, _, Dr = Al.shape
        Al = np.reshape(Al, [-1, Dr])
        q, r = np.linalg.qr(Al, mode="reduced")
        nrm = np.linalg.norm(r)
        # We remove the norms and store them in order to avoid overflow
        r = r/nrm
        norms.append(nrm)
        Dr = q.shape[-1]
        mpo[i] = np.reshape(q, [Dl, d, d, Dr])
        Al = np.einsum("ij,jklm->iklm", r, mpo[i+1])
        mpo[i+1] = Al

    # Compress MPO
    Ar = mpo[-1]
    eps = 1e-12
    dmax = 0
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
        dmax = max(dmax, Dl)
        # We bring back the norms to restore the norm of the entire MPO
        mpo[i] = np.reshape(v, [Dl, d, d, Dr])*norms[i-1]
        Ar = np.einsum("ijkl,lm,m->ijkm", mpo[i-1], u, s)
        mpo[i-1] = Ar
    print("Max bond dimension of the MPO:", dmax)
    return mpo


def compress_mpo(mpo):
    # Compress MPO
    Ar = mpo[-1]
    eps = 1e-12
    n = len(mpo)
    for i in range(n-1, 0, -1):
        Dl, d, _, Dr = Ar.shape
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
    # return mpo


def anonimize_mpo(mpo):
    n = len(mpo)
    nrm = np.linalg.norm(mpo[0])
    new_nrm = nrm**(1.0/n)
    mpo[0] = (mpo[0]/nrm)*new_nrm
    for i in range(n-1):
        Al = mpo[i]
        Ar = mpo[i+1]
        D = Ar.shape[0]
        Ol = ortho_group.rvs(D)
        Or = Ol.T
        Al = np.einsum("...i,ij->...j", Al, Ol)
        Ar = np.einsum("ij,j...->i...", Or, Ar)
        mpo[i] = Al*new_nrm
        mpo[i+1] = Ar
    mpo[n-1] = mpo[n-1]*new_nrm


def TFIM(nx, ny, J, hx, hz, permutation=None):
    Jz = []
    n = nx*ny
    if permutation is None:
        permutation = range(n)
    for ix in range(nx):
        for iy in range(ny):
            i = ix*ny+iy
            pi = permutation[i]
            if ix < nx-1:
                j = (ix+1)*ny+iy
                pj = permutation[j]
                Jz.append([pi, pj, J])
            if iy < ny-1:
                j = ix*ny+(iy+1)
                pj = permutation[j]
                Jz.append([pi, pj, J])
    Jz = np.array(Jz)
    # Since the fields are uniform we do not need to permute them
    hz = np.ones(n)*hz
    hx = np.ones(n)*hx
    return Jz, hz, hx, permutation


def Wishart(n, alpha, seed=None, shuffle=False, permutation=None, dtype=np.float64):
    m = int(alpha*n)
    S = np.sqrt(n/(n-1))*(np.eye(n)-np.ones([n, n])/n)
    rng = np.random.default_rng(seed)
    W = rng.multivariate_normal(
        np.zeros(n), np.eye(n), m, method='cholesky') @ S

    J = W.T @ W / n / alpha
    J = J - np.diag(np.diag(J))
    E0 = np.sum(J)

    gs_sol = np.ones(n)

    if shuffle:
        S = np.diag(np.sign(0.5-rng.uniform(size=n)))
        J = S @ J @ S
        gs_sol = np.diag(S)

    if permutation is None:
        permutation = range(n)

    gs_sol = gs_sol[permutation]
    J = J[permutation, :]
    J = J[:, permutation]

    # Inside the function
    assert is_symmetric(J), "Non symmetric input matrix J: stop"
    check_on_diagonal_J = not any(np.diag(J) != 0)
    assert check_on_diagonal_J, "Some diagonal elements of J are non zero: stop"

    Jz = []
    for i in range(n):
        for j in range(i):
            Jz.append([j, i, 2*J[j, i]])
    Jz = np.array(Jz, dtype=dtype)
    hz = np.zeros(n, dtype=dtype)
    J = np.array(J, dtype=dtype)

    print("n, m, m/n, alpha, E0: ", n, m, m/n, alpha, E0)

    return Jz, hz, J, gs_sol


def flat_sx_H0(n, inits=None, ext=False):
    # H0
    A = np.zeros([4, 2, 2, 4], dtype=np.cdouble)
    ix = np.array([[1, 1], [1, 1]])/2.
    i2 = np.eye(2)
    A[0, :, :, 1] = -i2
    A[1, :, :, 1] = i2
    A[1, :, :, 3] = i2
    A[0, :, :, 2] = 2*ix
    A[2, :, :, 2] = ix
    A[2, :, :, 3] = ix
    K = 1
    if ext:
        K = n
    if inits is not None:
        Am = np.zeros([4, 2, 2, 4], dtype=np.cdouble)
        ixm = np.array([[1, -1], [-1, 1]])/2.
        Am[0, :, :, 1] = -i2
        Am[1, :, :, 1] = i2
        Am[1, :, :, 3] = i2
        Am[0, :, :, 2] = 2*ixm
        Am[2, :, :, 2] = ixm
        Am[2, :, :, 3] = ixm
        mpo0 = []
        for s in inits:
            if s > 0:
                mpo0.append(A)
            else:
                mpo0.append(Am)
        mpo0[0] = mpo0[0][:1]*K
        mpo0[n-1] = mpo0[n-1][:, :, :, -1:]
    else:
        mpo0 = [A[:1, :, :, :]*K] + [A]*(n-2) + [A[:, :, :, -1:]]
    return mpo0


def search_mpos(n, state):
    # n is the number of spins
    # state is the designed spin configuration
    mpo0 = flat_sx_H0(n)

    # H1
    A = np.zeros([4, 2, 2, 4], dtype=np.cdouble)
    A[0, :, :, 1] = i2
    A[1, :, :, 1] = i2
    A[1, :, :, 3] = i2
    mpo1 = []
    for s in state:
        Ai = A.copy()
        Ai[0, s, s, 2] = -1.0
        Ai[2, s, s, 2] = 1.0
        Ai[2, s, s, 3] = 1.0
        mpo1.append(Ai)
    mpo1[0] = mpo1[0][:1]
    mpo1[-1] = mpo1[-1][:, :, :, -1:]

    return mpo0, mpo1
