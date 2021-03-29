# %% Package Imports

from dask.distributed import Client, progress
from dask_ml.feature_extraction.text import CountVectorizer

# %% Rewrite the basic functions in dask.

client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
# client


