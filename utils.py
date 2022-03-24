from scipy.sparse import csr_matrix
from numba import jit

@jit
def to_matrix(data, description):
    # transform dataframe into matrix 
    user_ids, item_ids = data[description['users']], data[description['items']]
    
    ratings = data[description['feedback']]
    shape = (description['n_users'], description['n_items'])
    
    return csr_matrix((ratings, (user_ids, item_ids)), shape)

@jit
def count_new_users_and_items(data_base_current, data_future_current):
    user_ids_base, item_ids_base = data_base_current['userid'].unique(), data_base_current['itemid'].unique()
    
    n_new_users = data_future_current[
        ~data_future_current['userid'].isin(user_ids_base)
    ]['userid'].nunique()
    
    n_new_items = data_future_current[
        ~data_future_current['itemid'].isin(item_ids_base)
    ]['itemid'].nunique()
    
    return n_new_users, n_new_items