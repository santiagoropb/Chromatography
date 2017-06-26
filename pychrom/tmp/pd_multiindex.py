# import modules
import pandas as pd
import numpy as np

# Create dataframe
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
        'preTestScore': [4, 24, np.nan, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns=['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])

print(df)

# set index to be multiindex
midf = df.set_index(['regiment', 'company'])

# set a value with the multiindex
midf.set_value(('Scouts', '1st'), 'preTestScore', 100)

raw_data2 = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
        'name': [np.nan, 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
        'preTestScore': [np.nan, 24, np.nan, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [np.nan, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df2 = pd.DataFrame(raw_data2, columns=['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
midf2 = df2.set_index(['regiment', 'company'])

# getting one complete column is the same
col = midf2['name'] # this will give a multiindex series
print(col)

# get Series # gets all cols for one particular multiindex
print(midf2.loc['Scouts','1st'])

# get dataframe (sub)
midf2.loc['Scouts', :]


# dropna(how='all',axis=0) drop all nans in rows. This will quickly reshape the data structure
# can use swaplevels if I want to change the order of the indexes
# maybe use stack and unstack for easier manipulation after droping stuff
# https://pandas.pydata.org/pandas-docs/stable/advanced.html
# https://chrisalbon.com/python/pandas_hierarchical_data.html
# Nelli book


