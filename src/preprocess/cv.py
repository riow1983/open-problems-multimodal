comp_name = "open-problems-multimodal"
dtype = 'multiome'
nb_name = f'localsrc002-{dtype}-cv'
col_sample = None
row_sample = 1000
    

import sys
import os
from pathlib import Path

KAGGLE_ENV = True if 'KAGGLE_URL_BASE' in set(os.environ.keys()) else False

if KAGGLE_ENV:
    BASE_DIR = Path('/kaggle/working')
elif "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}")
else:
    BASE_DIR = Path("/home/jovyan/kaggle")



INPUT_DIR = BASE_DIR / 'input'
# !mkdir {INPUT_DIR}
os.makedirs(INPUT_DIR, exist_ok=True)

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    OUTPUT_DIR = INPUT_DIR / nb_name
    # !mkdir {OUTPUT_DIR}
    os.makedirs(OUTPUT_DIR, exist_ok=True)



import pandas as pd
import gc
import h5py
import hdf5plugin
import numpy as np
from datareader import *

# DATA_DIR = "/home/jovyan/kaggle/input/open-problems-multimodal/"
DATA_DIR = INPUT_DIR / comp_name
# FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")
FP_CELL_METADATA = DATA_DIR / "metadata.csv"

# FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
# FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
# FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")
FP_CITE_TEST_INPUTS = DATA_DIR / "test_cite_inputs.h5"

# FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multi_inputs.h5"
# FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multi_targets.h5"
# FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")
FP_MULTIOME_TEST_INPUTS = DATA_DIR / "test_multi_inputs.h5"

# FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_SUBMISSION = DATA_DIR / "sample_submission.csv"
# FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")
FP_EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"

if dtype == 'cite':
    PATH_TO_X = FP_CITE_TRAIN_INPUTS
    PATH_TO_XT = FP_CITE_TEST_INPUTS
elif dtype == 'multiome':
    PATH_TO_X = FP_MULTIOME_TRAIN_INPUTS
    PATH_TO_XT = FP_MULTIOME_TEST_INPUTS
else:
    raise ValueError('dtype must be "cite" or "multiome"')




print('Now here!!!!!!!!!!!! (1)')
drx = DataReader(data_dir = PATH_TO_X.parent,
                 filename = PATH_TO_X.name,
                 metadata_file_name = 'metadata.csv')
X = drx.query_data(col_sample = col_sample, row_sample = row_sample)
del drx; gc.collect()
# iterX = pd.read_hdf(PATH_TO_X, iterator=True, chunksize=16_780)
# print('type of PATH_TO_X: ', type(PATH_TO_X))
# print(PATH_TO_X.name)
# f = h5py.File(PATH_TO_X, 'r')
# print(list(f.keys()))
# print(PATH_TO_X.name.split('.')[0])
# print(list(f[PATH_TO_X.name.split('.')[0]].keys()))
# X = f[PATH_TO_X.name.split('.')[0]]['block0_values']
# # X = hf.get(PATH_TO_X.name.split('.')[0]).value
# # X = X[:]
# X = np.array(X)
print(type(X))
print(X.shape)
# print(X)
len_X = len(X)


drxt = DataReader(data_dir = PATH_TO_XT.parent,
                  filename = PATH_TO_XT.name,
                  metadata_file_name = 'metadata.csv')
Xt = drxt.query_data(col_sample = col_sample, row_sample = row_sample) 
del drxt; gc.collect()
# # Xt = pd.read_hdf(PATH_TO_XT)
# f = h5py.File(PATH_TO_XT, 'r')
# Xt = f[PATH_TO_XT.name.split('.')[0]]['block0_values']
# Xt = np.array(Xt)
print(type(Xt))
print(Xt.shape)
# print(Xt)
print('Now here!!!!!!!!!!!! (2)')
# cols_full = Xt.columns
len_Xt = len(Xt)
# XXt = pd.concat([X, Xt], axis=0).reset_index(drop=True)
XXt = np.concatenate([X, Xt], axis=0)
print(XXt.shape)
print(XXt)
# for X in iterX:
#     print('X.shape before dropping columns: ', X.shape)
#     XXt = pd.merge(X, Xt, on='cell_id', how='inner')
#     print('XXt.shape before dropping columns: ', XXt.shape)
del X, Xt; gc.collect()
assert len(XXt) == len_X + len_Xt
print('Now here!!!!!!!!!!!! (3)')



    
        

# Detect and drop constant columns
# XXt = XXt.loc[:, (XXt != XXt.iloc[0]).any()]
# gc.collect()
# print('XXt.shape after dropping columns: ', XXt.shape)
# cols_remained = set(XXt.columns())
# cols_remained = cols_full[(XXt != XXt.iloc[0]).any()]

# cols_dropped = set(cols_full) - set(cols_remained)
# print('¥n¥n¥nNumber of columns dropped: ', len(cols_dropped))
# print('¥n¥n¥nColumns dropped: ¥n¥n', cols_dropped)

# X = XXt.iloc[:len_X, :]
# Xt = XXt.iloc[len_X:, :]
# del XXt: gc.collect()
# X.to_hdf(OUTPUT_DIR / 'X_multi.h5', key='X')
# Xt.to_hdf(OUTPUT_DIR / 'Xt_multi.h5', key='Xt')

# CV