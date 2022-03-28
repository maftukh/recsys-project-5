import numpy as np

from prepare_data import train_test_val_split
from prepare_data import get_random_user_subset, get_chronological_sorted_data
from amazon_loader import get_amazon_data
from utils import get_sparse_matrix

from svd import svd_col_expansion, svd_expansion, get_svd_model

from config import NUM_NEW_ROWS as num_new_rows
from config import NUM_NEW_COLS as num_new_cols
from config import RANK as rank

from config import SUBSET_SIZE as subset_size
from config import TRAIN_PART as train_prt
from config import TEST_PART as test_prt

import warnings
warnings.filterwarnings('ignore')

# Download data
data = get_amazon_data()

# Prepare data
# Get data subset. Just for making calculations faster
data = get_random_user_subset(data, size=subset_size)
data = get_chronological_sorted_data(data)

# Split data
train, val, test = train_test_val_split(data,
                                        train_size=train_prt,
                                        test_size=test_prt)

# prepare and slice data
train_matrix = get_sparse_matrix(train)

M = train_matrix[:, :-num_new_cols]
C = train_matrix[:, -num_new_cols:]
R = train_matrix[-num_new_rows:, :]

# apply svd
U, s, Vt = get_svd_model(M, rank, False)

print('U shape (before expansion):', U.shape)
print('Vt shape: (before expansion)', Vt.shape)

# apply expansion
U_new, s_new, Vt_new = svd_expansion(U, s, Vt, R, C)

print('U shape (after expansion):', U_new.shape)
print('Vt shape: (after expansion)', Vt_new.shape)

# Ortogonality check asserts
assert np.linalg.norm((U.T @ U) - np.eye(rank)) / np.sqrt(rank) < 10e-7, 'U^T @ U relative error is too large'
assert np.linalg.norm((Vt @ Vt.T) - np.eye(rank)) / np.sqrt(rank) < 10e-7, 'V^T @ V relative error is too large'
assert np.linalg.norm((U_new.T @ U_new) - np.eye(rank)) / np.sqrt(rank) < 10e-7, 'U_new^T @ U_new relative error is too large'
assert np.linalg.norm((Vt_new @ Vt_new.T) - np.eye(rank)) / np.sqrt(rank) < 10e-7, 'V_new^T @ V_new relative error is too large'
