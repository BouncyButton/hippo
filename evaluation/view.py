import pickle
import pandas as pd

# open adni_hippocampus_full.pkl file with gzip compression
d = pd.read_pickle("msd_hippocampus_full.pkl", compression="gzip")

print(d.head())