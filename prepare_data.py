import numpy as np
import warnings 
import pandas as pd

def get_random_user_subset(data, size, unique_users=None):
    if unique_users is None:
        unique_users = data['userid'].unique()
        
    user_subset = np.random.choice(unique_users, size=size)
    data = data[data['userid'].isin(user_subset)]
    
    return data

def get_chronological_sorted_data(data):
    # change user and item ids to be indexed in chronological order
    user_ids = data.sort_values('timestamp')['userid'].unique()
    user_map = dict(zip(user_ids, range(len(user_ids))))
    data['userid'] = data['userid'].map(user_map)

    item_ids = data.sort_values('timestamp')['itemid'].unique()
    item_map = dict(zip(item_ids, range(len(item_ids))))
    data['itemid'] = data['itemid'].map(item_map)
    
    return data

def base_future_split(data, base_size=0.5):
    ts_base = data['timestamp'].quantile(base_size)

    data_base = (
        data
        .query(f"timestamp <= @ts_base")
        .sort_values('timestamp')
    )
    
    data_future = (
        data
        .query(f"timestamp > @ts_base")
        .sort_values('timestamp')
    )
    return data_base, data_future

def get_holdout_subset(data_base, data_future):
    # get first interaction for each user in validation set
    data_holdout = data_future.drop_duplicates(subset='userid', keep='first')
    
    # remove completely new users or items
    user_ids = data_base['userid'].unique()
    data_holdout = data_holdout.query("userid in @user_ids")
    
    item_ids = data_base['itemid'].unique()
    data_holdout = data_holdout.query("itemid in @item_ids")
    
    if data_holdout.shape[0] == 0:
        warnings.warn("WARNING: holdout has no valid user for predicting")
        
    return data_holdout

def get_future_subsets(data_future, n_steps=20):
    ts_min, ts_max = data_future['timestamp'].min(), data_future['timestamp'].max() + 1
    checkpoints = np.linspace(ts_min, ts_max, n_steps + 1)
    
    splits = []
    start_times = checkpoints[:-1]
    stop_times = np.roll(checkpoints, -1)[:-1]
    
    for start, stop in zip(start_times, stop_times):
        split = data_future.query(f"timestamp >= @start and timestamp < @stop")
        
        if split.shape[0] == 0:
            warnings.warn("WARNING: validation split time period has no observation")
        
        splits.append(split)
    
    return splits, start_times, stop_times

def get_updated_base(data_base, data_future):
    return pd.concat([data_base, data_future])

def train_test_val_split(data, train_size=0.5, test_size=0.2):
    assert train_size < 1 - test_size, "test data should not overlap train data"

    train_ts = data['timestamp'].quantile(train_size)
    test_ts = data['timestamp'].quantile(1 - test_size)

    train = data.query(f"timestamp <= @train_ts")
    val = data.query(f"timestamp > @train_ts and timestamp <= @test_ts")
    val = val.sort_values('timestamp')
    test = data.query(f"timestamp > @test_ts")
    return train, val, test
