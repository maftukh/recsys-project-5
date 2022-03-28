from scipy.sparse import csr_matrix, coo_matrix
from numba import jit

# Transform dataframe into matrix
@jit
def to_matrix(data, description):
    # transform dataframe into matrix 
    user_ids, item_ids = data[description['users']], data[description['items']]
    
    ratings = data[description['feedback']]
    shape = (description['n_users'], description['n_items'])
    
    return csr_matrix((ratings, (user_ids, item_ids)), shape)

# Count new users and items
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

# Convert dataframe to sparse matrix
@jit
def get_sparse_matrix(data, sparse_format='csr'):
    # Extract required data
    user_idx = data['userid'].values
    item_idx = data['itemid'].values
    ratings = data['rating'].values

    n_users = data['userid'].nunique()
    n_items = data['itemid'].nunique()
    shape = (n_users, n_items)
        
    # Create a sparse user-item interaction matrix of specified format
    sparse_matrix_foos = {
        'csr': csr_matrix,
        'coo': coo_matrix
    }
    sparse_foo = sparse_matrix_foos[sparse_format]
    user_item_mtx = sparse_foo((ratings, (user_idx, item_idx)), shape=shape, dtype='float64')
    return user_item_mtx
   
# Create description as dictionary
@jit
def get_description_dict(data):
    description = {
    'users': 'userid',
    'items': 'itemid',
    'feedback': 'rating',
    
    'n_users': data['userid'].nunique(),
    'n_items': data['itemid'].nunique()
    }

    return description
