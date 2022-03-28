import numpy as np
from numpy.linalg import qr, svd

from scipy.sparse.linalg import svds
from numba import jit

from prepare_data import get_holdout_subset, get_updated_base
from evaluation import model_evaluate, downvote_seen_items, simple_model_recom_func
from utils import to_matrix, count_new_users_and_items

import time
from tqdm import tqdm

from config import RANK as rank
from config import N_TOP as n_top
from config import N_STEPS as n_steps

# Get SVD sparse matrix decomposition
@jit
def get_svd_model(matrix, rank, ascontiguous=False):
    u, s, vt = svds(matrix, k=rank)
    
    if ascontiguous:
        U = np.ascontiguousarray(u[:, ::-1])
        V = np.ascontiguousarray(vt[::-1, :].T)
        s = s[::-1]
        params = U, s, V
    else:
        params = u, s, vt
        
    return params

# Get scores of SVD model
@jit
def score_svd_model(params, matrix):
    _, _, V = params
    scores = matrix @ V @ V.T
    
    return scores

# Update SVD factors using new interactions
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

# Update SVD model using new users and items information
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

# Apply base SVD approach
@jit
def baseline_svd(splits, data_base_current, matrix_base_current, params_current, description, verbose=False):
    basic_performance_history = []
    basic_time_history = []
    for split in tqdm(splits):
        
        ## ------------------------------------------- prepare data -------------------------------------------
        
        # get current valid
        data_future_current = split
        
        # extract holdout from valid
        data_holdout = get_holdout_subset(data_base_current, data_future_current)
        
        # prepare matrix for holdout users
        user_ids_holdout = data_holdout['userid'].values
        matrix_observed = matrix_base_current[user_ids_holdout]
        
        # prepare observed in order to downvote after scoring
#        data_observed = data_base_current.query('userid in @user_ids_holdout')
        
        data_observed = data_base_current[data_base_current.userid.isin(user_ids_holdout)]
        
        # prepare holdout and observed for evaluating — reindex users
        reindex_map = dict(zip(user_ids_holdout, np.arange(len(user_ids_holdout))))
        
        for _ in [data_holdout, data_observed]:
            _.loc[:, 'userid'] = _['userid'].map(reindex_map)
            
        ## ---------------------------------- perform scoring and evaluation ----------------------------------

        # score holdout users
        scores = score_svd_model(params_current, matrix_observed)
        if verbose: print('scoring done!')
        
        # downvote items which occured on train
        downvote_seen_items(scores, data_observed, description)
        
        # construct predictions
        predictions = simple_model_recom_func(scores, topn=n_top)
        if verbose: print('predictions done!')

        # evaluate metrics on holdout scores
        metrics = model_evaluate(predictions, data_holdout, description, topn=n_top)
        basic_performance_history.append(metrics)
        if verbose: print('metrics done!')
            
        ## -------------------------------------- update model parameters -------------------------------------
        
        # update train as data
        data_base_current = get_updated_base(data_base_current, data_future_current)
        
        description['n_users'] = data_base_current['userid'].nunique()
        description['n_items'] = data_base_current['itemid'].nunique()
        
        # recompute SVD factors of train data
        start = time.time()
        matrix_base_current = to_matrix(data_base_current, description)
        params_current = get_svd_model(matrix_base_current, rank, True)
        end = time.time()
        
        basic_time_history.append(end - start)
        if verbose: print('update done!')

    basic_performance_history = np.array(basic_performance_history)
    basic_time_history = np.array(basic_time_history)

    return params_current, basic_performance_history, basic_time_history

# Apply dynamic SVD approach
@jit
def dynamic_svd(splits, data_base_current, matrix_base_current, params_current, description, verbose=False):
    dynamic_performance_history = []
    dynamic_time_history = []

    verbose = False
    for split in tqdm(splits):
        
        ## ------------------------------------------- prepare data -------------------------------------------
        
        # get current valid
        data_future_current = split
        
        # extract holdout from valid
        data_holdout = get_holdout_subset(data_base_current, data_future_current)
        
        # prepare matrix for holdout users
        user_ids_holdout = data_holdout['userid'].values
        matrix_observed = matrix_base_current[user_ids_holdout]
        
        # prepare observed in order to downvote after scoring
#        data_observed = data_base_current.query('userid in @user_ids_holdout')
        data_observed = data_base_current[data_base_current.userid.isin(user_ids_holdout)]
        
        
        # prepare holdout and observed for evaluating — reindex users
        reindex_map = dict(zip(user_ids_holdout, np.arange(len(user_ids_holdout))))
        
        for _ in [data_holdout, data_observed]:
            _.loc[:, 'userid'] = _['userid'].map(reindex_map)
            
        ## ---------------------------------- perform scoring and evaluation ----------------------------------

        # score holdout users
        scores = score_svd_model(params_current, matrix_observed)
        if verbose: print('scoring done!')
        
        # downvote items which occured on train
        downvote_seen_items(scores, data_observed, description)
        
        # construct predictions
        predictions = simple_model_recom_func(scores, topn=n_top)
        if verbose: print('predictions done!')

        # evaluate metrics on holdout scores
        metrics = model_evaluate(predictions, data_holdout, description, topn=n_top)
        dynamic_performance_history.append(metrics)
        if verbose: print('metrics done!')
            
        ## -------------------------------------- update model parameters -------------------------------------
            
        # count number of new users and items
        n_new_users, n_new_items = count_new_users_and_items(data_base_current, data_future_current)
        
        # update SVD factors using new interactions
        start = time.time()
        data_new = data_future_current[
            data_future_current['userid'].isin(data_base_current['userid'].unique()) &
            data_future_current['itemid'].isin(data_base_current['itemid'].unique())
        ]
        matrix_new = to_matrix(data_new, description)
        params_current = update_svd_model_using_interactions(params_current, matrix_new)
        end = time.time()
        
        dynamic_time_history.append(end - start)
        
        # update train as data
        data_base_current = get_updated_base(data_base_current, data_future_current)

        description['n_users'] = data_base_current['userid'].nunique()
        description['n_items'] = data_base_current['itemid'].nunique()
        
        # update SVD factors using new users and items
        start = time.time()
        matrix_base_current = to_matrix(data_base_current, description)
        
        rows_new, columns_new = matrix_base_current[-n_new_users:, :], matrix_base_current[:, -n_new_items:]
        params_current = update_svd_model_using_users_and_items(params_current, rows_new, columns_new)
        end = time.time()
        
        dynamic_time_history[-1] += end - start
        if verbose: print('update done!')

    dynamic_performance_history = np.array(dynamic_performance_history)
    dynamic_time_history = np.array(dynamic_time_history)

    return params_current, dynamic_performance_history, dynamic_time_history

# SVD columns expansion
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

# General SVD expansion
@jit
def svd_expansion(U, s, Vt, R, C):
    U_new = U.copy()
    s_new = s.copy()
    Vt_new = Vt.copy()
    if C.shape[1] > 0:
        U_new, s_new, Vt_new = svd_col_expansion(U_new, s_new, Vt_new, C)
    if R.shape[0] > 0:
        Vt_new, s_new, U_new = svd_col_expansion(Vt.T, s_new, U.T, R.T)
        Vt_new = Vt_new.T
        U_new = U_new.T
    return U_new, s_new, Vt_new
