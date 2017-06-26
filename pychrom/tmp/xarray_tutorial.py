import pandas as pd
import numpy as np
import xarray as xr

data = np.random.rand(4, 3, 2)
locs = ['IA', 'IL', 'IN']
times = range(4)
components = ['A', 'B']
foo = xr.DataArray(data, coords=[times, locs, components], dims=['time', 'space', 'components'])

# positional
# gets all space and components for time 0 and 1
print(foo.loc[0:1])
# gets all components for time 1 and space 1
print(foo.loc[1][1])
# gets all components for time 1, space 1 and 2
print(foo.loc[1][0:2])

data = np.random.rand(4, 3, 2)
locs2 = np.linspace(0, 2, 3)
times2 = np.linspace(0, 10, 4)
components = ['A', 'B']
foo2 = xr.DataArray(data, coords=[times2, locs2, components], dims=['time', 'space', 'components'])

# links
# http://xarray.pydata.org/en/stable/whats-new.html