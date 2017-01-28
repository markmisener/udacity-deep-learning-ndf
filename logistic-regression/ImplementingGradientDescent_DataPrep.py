import io
import numpy as np
import pandas as pd
import requests

# pull data from url into CSV
url = 'http://www.ats.ucla.edu/stat/data/binary.csv'
response = requests.get(url).content
data = pd.read_csv(io.StringIO(response.decode('utf-8')))

# add columns with binary values for ranks and drop the original rank column
data = pd.concat([data,pd.get_dummies(data['rank'], prefix='rank')], axis=1)
data.drop('rank', axis=1, inplace=True)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
