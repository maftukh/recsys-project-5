import numpy as np
from scipy.sparse.linalg import svds 
from numba import jit

@jit
def get_svd_model(matrix, rank):
    u, s, vt = svds(matrix, k=rank)
    
    U = np.ascontiguousarray(u[:, ::-1])
    V = np.ascontiguousarray(vt[::-1, :].T)
    s = s[::-1]
    
    params = U, s, V
    return params

@jit
def score_svd_model(params, matrix):
    _, _, V = params
    scores = matrix @ V @ V.T
    
    return scores

@jit
def update_svd_model_using_interactions(params_base, A_new, verbose=False):
    U_base, s_base, V_base = params_base
    S_base = np.diag(s_base)
    
    # m is number of users and n is number of items
    
    temp = A_new @ V_base # m x r
    rank = len(s_base)
    
    # 1st step
    K = U_base @ S_base + temp # m x r
    if verbose: print('1st step done!')
    
    # 2nd step
    U_updated, _ = qr(K) # m x r, r x r
    if verbose: print('2nd step done!')
    
    # 3rd step
    _ = _ - U_updated.T @ temp # r x r
    if verbose: print('3rd step done!')
    
    # 4th step
    L = V_base @ _.T + A_new.T @ U_updated # n x r
    if verbose: print('4th step done!')
    
    # 5th step
    V_updated, S_updated = qr(L) # n x r, r x r
    if verbose: print('5th step done!')
    
    params_updated = U_updated, np.diag(S_updated), V_updated
    return params_updated

@jit
def svd_col_expansion(U, s, Vt, C):
    rank = len(s)
    C = C[:U.shape[0]]

    L = U.T @ C
    H = C - U @ L
    J, K = qr(H)
    
    volume = np.sqrt(np.linalg.det(K.T @ K))

    Q = np.block([
        [np.diag(s),                     L],
        [np.zeros((K.shape[0], len(s))), K]
    ])
    
    U_q, s_q, Vt_q = svd(Q, full_matrices=True)

    U_new = np.hstack((U, J)) @ U_q
    s_new = s_q
    
    Vt_new = Vt_q @ np.block([
        [Vt,                                  np.zeros((Vt.shape[0], C.shape[1]))],
        [np.zeros((C.shape[1], Vt.shape[1])), np.eye(C.shape[1])]
    ])

    return U_new[:,:rank], s_new[:rank], Vt_new[:rank,:]

@jit
def update_svd_model_using_users_and_items(params_base, R, C):
    
    if C.shape[1] > 0:
        U_base, s_base, V_base = params_base
        _, s_new, Vt_new = svd_col_expansion(U_base, s_base, V_base.T, C)
        V_new = Vt_new.T
    
    if R.shape[0] > 0:
        U_base, s_base, V_base = params_base
        _, s_new, Ut_new = svd_col_expansion(V_base, s_new, U_base.T, R.T)
        U_new = Ut_new.T
    
    return U_new, s_new, V_new