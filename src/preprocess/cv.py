comp_name = "open-problems-multimodal"
dtype = 'multiome'
nb_name = f'localsrc002-{dtype}-cv'
col_sample = None
row_sample = 1000
N_SPLITS = 3
import numpy as np
np.random.seed(42)
    

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
os.makedirs(INPUT_DIR, exist_ok=True)

if KAGGLE_ENV:
    OUTPUT_DIR = Path('')
else:
    OUTPUT_DIR = INPUT_DIR / nb_name
    os.makedirs(OUTPUT_DIR, exist_ok=True)



import pandas as pd
import gc
import h5py
import hdf5plugin
from sklearn.model_selection import GroupKFold
from datareader import *

if "google.colab" in sys.modules:
    python_version = 7
elif KAGGLE_ENV:
    python_version = 7
else:
    python_version = 10
if not os.path.exists(f'/opt/conda/lib/python3.{str(python_version)}/site-packages/tables'):
    os.system('pip install --quiet tables')

DATA_DIR = INPUT_DIR / comp_name
FP_CELL_METADATA = DATA_DIR / "metadata.csv"
FP_CITE_TRAIN_INPUTS = DATA_DIR / "train_cite_inputs.h5"
FP_CITE_TRAIN_TARGETS = DATA_DIR / "train_cite_targets.h5"
FP_CITE_TEST_INPUTS = DATA_DIR / "test_cite_inputs.h5"
FP_MULTIOME_TRAIN_INPUTS = DATA_DIR / "train_multi_inputs.h5"
FP_MULTIOME_TRAIN_TARGETS = DATA_DIR / "train_multi_targets.h5"
FP_MULTIOME_TEST_INPUTS = DATA_DIR / "test_multi_inputs.h5"
FP_SUBMISSION = DATA_DIR / "sample_submission.csv"
FP_EVALUATION_IDS = DATA_DIR / "evaluation_ids.csv"

if dtype == 'cite':
    PATH_TO_X = FP_CITE_TRAIN_INPUTS
    PATH_TO_XT = FP_CITE_TEST_INPUTS
    PATH_TO_Y = FP_CITE_TRAIN_TARGETS 
elif dtype == 'multiome':
    PATH_TO_X = FP_MULTIOME_TRAIN_INPUTS
    PATH_TO_XT = FP_MULTIOME_TEST_INPUTS
    PATH_TO_Y = FP_MULTIOME_TRAIN_TARGETS
else:
    raise ValueError('dtype must be "cite" or "multiome"')


print(
'''
##################################################
#             Read Sampled Data                  #
##################################################
'''
)

# X
drx = DataReader(data_dir = PATH_TO_X.parent,
                 filename = PATH_TO_X.name,
                 metadata_file_name = 'metadata.csv')
X, row_sample_idx = drx.query_data(col_sample = col_sample, row_sample = row_sample)
del drx; gc.collect()
# len_X = len(X)


# Xt
drxt = DataReader(data_dir = PATH_TO_XT.parent,
                  filename = PATH_TO_XT.name,
                  metadata_file_name = 'metadata.csv')
Xt, _ = drxt.query_data(col_sample = col_sample, row_sample = row_sample) 
del drxt, _; gc.collect()
# len_Xt = len(Xt)


# # XXt (= X ; Xt)
# XXt = np.concatenate([X, Xt], axis=0)
# del X, Xt; gc.collect()
# assert len(XXt) == len_X + len_Xt


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



print('''
##################################################
#                      CV                        #
##################################################
''')

metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology==dtype]

cell_index = Xt.index
meta = metadata_df.reindex(cell_index)


# Read Y
Y = pd.read_hdf(PATH_TO_Y)
print(Y.shape)
Y = Y.iloc[row_sample_idx, :]
print(Y.shape)
y_columns = list(Y.columns)
Y = Y.values

# Normalize the targets row-wise: This doesn't change the correlations,
# and negative_correlation_loss depends on it
Y -= Y.mean(axis=1).reshape(-1, 1)
Y /= Y.std(axis=1).reshape(-1, 1)


# Cross-validation
kf = GroupKFold(n_splits=N_SPLITS)
X.reset_index(inplace=True)
Y = pd.DataFrame(Y)
X['fold'] = -1
X = pd.concat([X, Y], axis=1)
del Y; gc.collect()

for n, (train_index, val_index) in enumerate(kf.split(X, groups=meta.donor)):
    X.loc[val_index, 'fold'] = int(n)
X['fold'] = X['fold'].astype(int)
X.set_index('cell_id', inplace=True)
del metadata_df, meta, cell_index; gc.collect()


print('''
##################################################
#               Export Datasets                  #
##################################################
''')
X.to_hdf(OUTPUT_DIR / f'X_{dtype}_rs{row_sample}_fold.h5', key='X')
Xt.to_hdf(OUTPUT_DIR / f'Xt_{dtype}_rs{row_sample}_fold.h5', key='Xt')
