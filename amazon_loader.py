import gzip
import os
import requests
import pandas as pd
from ast import literal_eval

# Reader of Amazon data
def amazon_data_reader(path):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield literal_eval(line)

# Download and read data to dataframe
def get_amazon_data(json_filename = 'reviews_Electronics_5.json.gz',
                    url_json = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'):
    
    if json_filename in os.listdir():
        pass
    else:
        r = requests.get(url_json)
        with open(json_filename , 'wb') as f:
            f.write(r.content)

    col_names_mapping = dict(zip(
        ['reviewerID', 'asin', 'overall', 'unixReviewTime'],
        ['userid', 'itemid', 'rating', 'timestamp']
    ))

    pure_data = pd.DataFrame.from_records(
        amazon_data_reader(json_filename),
        columns=['reviewerID', 'asin', 'overall', 'unixReviewTime']
    ).rename(columns=col_names_mapping)

    return pure_data
