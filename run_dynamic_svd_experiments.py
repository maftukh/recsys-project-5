from svd import get_svd_model, score_svd_model, update_svd_model_using_interactions, svd_col_expansion, update_svd_model_using_users_and_items
from svd import baseline_svd, dynamic_svd

from evaluation import model_evaluate, simple_model_recom_func, downvote_seen_items

from amazon_loader import get_amazon_data
from prepare_data import get_random_user_subset, get_chronological_sorted_data, base_future_split
from prepare_data import get_holdout_subset, get_future_subsets, get_updated_base

from utils import to_matrix, count_new_users_and_items, get_description_dict

from config import RANK as rank
from config import N_TOP as n_top
from config import N_STEPS as n_steps
from config import SUBSET_SIZE as subset_size
from config import BASE_PART as base_part

from visualization import plot_metric

import warnings
warnings.filterwarnings('ignore')

# Download data
pure_data = get_amazon_data()

# Prepare data
# Get data subset. Just for making calculations faster
data = get_random_user_subset(pure_data, size=subset_size)
data = get_chronological_sorted_data(data)

# Split data (base-future subsets)
data_base, data_future = base_future_split(data, base_size=base_part)

# ============= BASE SVD PART =============

# Description dictionary
description = get_description_dict(data_base)

# Convert data to matrix format
data_base_current = data_base.copy()
matrix_base_current = to_matrix(data_base_current, description)

# Apply svd decomposition
params_current = get_svd_model(matrix_base_current, rank, True)

# Split future data on folds
splits, start_times, stop_times = get_future_subsets(data_future, n_steps)

print('Basic SVD computation progress:')
params_current, basic_performance_history, basic_time_history = baseline_svd(splits,
                                                                             data_base_current,
                                                                             matrix_base_current,
                                                                             params_current,
                                                                             description,
                                                                             verbose=False)
                                                                             
# ============= DYNAMIC SVD PART =============

# Description dictionary
description = get_description_dict(data_base)

# Convert data to matrix format
data_base_current = data_base.copy()
matrix_base_current = to_matrix(data_base_current, description)

# Apply svd decomposition
params_current = get_svd_model(matrix_base_current, rank, True)

# Split future data on folds
splits, start_times, stop_times = get_future_subsets(data_future, n_steps)

print('Dynamic SVD computation progress:')
params_current, dynamic_performance_history, dynamic_time_history = dynamic_svd(splits,
                                                                                data_base_current,
                                                                                matrix_base_current,
                                                                                params_current,
                                                                                description,
                                                                                verbose=False)
                                                                                
# Plot and compare results

# HR metric plot
plot_metric('HR',
            stop_times,
            basic_performance_history[:, 0],
            dynamic_performance_history[:, 0],
            n_top=n_top)

# MRR metric plot
plot_metric('MRR',
            stop_times,
            basic_performance_history[:, 1],
            dynamic_performance_history[:, 1],
            n_top=n_top)

# Coverage metric plot
plot_metric('coverage',
            stop_times,
            basic_performance_history[:, 2],
            dynamic_performance_history[:, 2],
            n_top=n_top)

# Times plot
plot_metric('time per one iteration, sec',
            stop_times,
            basic_time_history,
            dynamic_time_history,
            n_top=n_top)
