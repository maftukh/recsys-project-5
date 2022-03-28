import numpy as np
import pandas as pd

# Evaluate model metrics
def model_evaluate(recommended_items, holdout, holdout_description, topn=10, to_return=['HR', 'MRR', 'coverage']):
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    
    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))
    
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    
    # DCG calculation
    dcg = np.sum(1 / np.log2(1 + hit_rank)) / n_test_users
    
    # coverage calculation
    n_items = holdout_description['n_items']
    coverage = np.unique(recommended_items).size / n_items
    
    metrics = []
    if 'HR' in to_return:
        metrics.append(hr)
    
    if 'MRR' in to_return:
        metrics.append(mrr)
    
    if 'DCG' in to_return:
        metrics.append(dcg)
        
    if 'coverage' in to_return:
        metrics.append(coverage)
        
    return metrics

# Get best simple recommendations
def simple_model_recom_func(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

# Get best recommendations indexes
def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]

# Downvote already seen items
def downvote_seen_items(scores, data, data_description):
    itemid = data_description['items']
    userid = data_description['users']
    
    # get indices of observed data
    sorted = data.sort_values(userid)
    item_idx = sorted[itemid].values
    user_idx = sorted[userid].values
    user_idx = np.r_[False, user_idx[1:] != user_idx[:-1]].cumsum()
    
    # downvote scores at the corresponding positions
    seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores.shape)
    np.put(scores, seen_idx_flat, scores.min() - 1)
