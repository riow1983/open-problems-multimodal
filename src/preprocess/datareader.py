# Credit to https://www.kaggle.com/code/alekeuro/querying-specific-subsets-of-the-data-without-load/notebook
import os
if not os.path.exists('/opt/conda/lib/python3.10/site-packages/tables'):
    os.system('pip install --quiet tables')
os.system('pip install hdf5plugin')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import hdf5plugin
import tables
import psutil
import time
import gc
import pathlib

class DataReader:
    """
    Class to access subsets of data in one of the *.h5 files, without loading all of the data into memory.
    E.g., if you are working in a kaggle notebook, and want to read multiome training inputs, initialize a 
    data reader like:
    dr = DataReader(data_dir = '/kaggle/input',
                    filename = 'training_multi_inputs.h5',
                    metadata_file_name = 'metadata.csv')
                    
    Use `DataReader.query_data` method to query the data.
    
    GOOD TO KNOW: if you want to query data from, say, donor 31800, you can use donor_id = 2. Example:
    dr.query_data(donor_id = 2) --> returns all data available from donor = 31800. 
    The mapping between donor and donor_id is given below:
    
    donor | donor_id
    ----------------
    13176 | 1
    31800 | 2
    32606 | 3
    27678 | 4
    
    Note: the donor_id identifier is set to 4 for donor 27678 (instead of 2) in order to be in agreement with
    Data documentation provided by the organizers (donor 27678 is informally called "Donor 4" in the Data tab 
    of the competition's main page, and all around).
    
    
    """
    
    def __init__(self, data_dir, filename, metadata_file_name):
        self.filename = filename
        self.prefix = filename.replace('.h5','')
        self.file_path = os.path.join(data_dir, filename)
        self.md = pd.read_csv(os.path.join(data_dir, metadata_file_name))
        self.add_donor_id(self.md)
        
    def add_donor_id(self, df):

        donor_dict = {13176:1, 31800:2,32606:3,27678:4}
        df['donor_id'] = df.donor.map(donor_dict)
        
        
    def get_query_metadata(self, **kwargs):
        """
        Filters metadata according to variable query args (kwargs).
        """
        qmd = self.md.copy()
        for arg in kwargs:
            qmd = qmd.loc[qmd[arg] == kwargs[arg]].copy()
            
        return qmd
    
    @property
    def shape(self):
        with h5py.File(self.file_path,'r') as f:
            shape = f[self.prefix]['block0_values'].shape
        return shape
    
    def query_data(self, col_sample = None, row_sample = None, **kwargs):
        """
        This function uses h5py to query the file. It returns a dataframe (which might well be huge). 
        Depending on your query, you might still run out of RAM and your session crash.
        
        Inputs:
            - col_sample: int | list of string's | list of int's
                if col_sample is None (i.e., simply omitted), then returns all columns.
                if int: a random sample of `col_sample` columns will be chosen for the output
                if list of int's: these int's are interpreted as the desired columns positions.
                if list of string's: these must be column names.
                
                Warning: If you ask for a number larger than the number of columns, or you choose 
                         (string) column names that are not present in the actual columns, this 
                         will break, and not in a nice way.
                Remark: The output dataframe will have its columns ordered coherently with the actual
                        column ordering in the files. E.g., if you pass in col_sample = [5000, 14],
                        the first output column will be column nb. 14, and the second will be the 5000-th
                        columns.
                        
        kwargs: additional query args. These args must be chosen among the column names of the
                metadata.csv file.
                NOTE: there are four donors with four 5-digit codes. If you want to get data from 
                donor 13176, but you don't remember all these 5-digit codes, you can filter by donor_id.
                The donor <--> donor_id mapping is as follows:
                
                donor | donor_id
                ----------------
                13176 | 1
                31800 | 2
                32606 | 3
                27678 | 4                
                
                
        Output:
            - out: pandas.DataFrame with the desired rows (filtered according to kwargs) and columns.
            
        Examples:
        self.query_data(col_sample = 400, day = 2, donor_id = 1)
        Returns all rows for donor_id = 1 (i.e., donor = 13176), for day = 2, and 400 randomly sampled columns.
        self.query_data(col_sample = [3, 5, 1], day = 3, donor_id = 2, cell_type = 'BP')
        Returns all rows for donor_id = 2 (i.e., donor = 31800), for day = 3, and the second, fourth and sixth columns
        (in that order).
        
        """
        
        f = h5py.File(self.file_path, 'r')
        
        query_df = self.get_query_metadata(**kwargs)
        
        #### RIOW
        # rows = f[self.prefix]['axis1'][:]
        # rows = list(map(lambda x: x.decode(), rows))
        if row_sample == None: 
            N_rows = f[self.prefix]['axis1'].shape[0]
            row_sample_idx = np.arange(N_rows)
            rows = [row.decode() for row in f[self.prefix]['axis1'][:]]
        elif type(row_sample) == int: 
            N_rows = f[self.prefix]['axis1'].shape[0]
            row_sample_idx = np.random.choice(N_rows, size = row_sample, replace = False)
            row_sample_idx = sorted(list(row_sample_idx))
            rows = [row.decode() for row in f[self.prefix]['axis1'][row_sample_idx]]
        elif isinstance(row_sample, pathlib.PosixPath):
            rf = pd.read_csv(row_sample)
            cell_ids = rf['cell_id'].unique()
            N_rows = f[self.prefix]['axis1'].shape[0]
            row_sample_idx = np.arange(N_rows)
            rows = [row.decode() for row in f[self.prefix]['axis1'][:] if row.decode() in cell_ids]
        elif type(row_sample) == list:
            if np.max(row_sample) > 1.0: raise ValueError('Max of row_sample must be less than 1.0!')
            if np.min(row_sample) < 0.0: raise ValueError('Min of row_sample must be greater than or equal to 0.0!')
            N_rows = f[self.prefix]['axis1'].shape[0]
            if len(row_sample) != 2: 
                raise ValueError('row_sample must contain two elements!')
            else:
                if row_sample[1] < row_sample[0]: raise ValueError('The second element of row_sample must be larger than the first element of row_sample!')
            row_sample_idx = np.arange(np.floor(N_rows*row_sample[0]).astype(int), np.floor(N_rows*row_sample[1]).astype(int))
            rows = [row.decode() for row in f[self.prefix]['axis1'][row_sample_idx]]
        if len(rows)==0: raise ValueError('rows contains nothing!')
        print('rows[:5]:\n', rows[:5])
            
        #### RIOWRIOW
        rows = pd.DataFrame(rows, columns = ['cell_id']).reset_index().rename(columns = {'index':'index_col'})
        rows = pd.merge(query_df[['cell_id']], rows, on = 'cell_id').sort_values(by = 'index_col')
        
        if col_sample == None: 
            N_cols = f[self.prefix]['axis0'].shape[0]
            col_sample_idx = np.arange(N_cols)
            columns = [col.decode() for col in f[self.prefix]['axis0'][:]]
        elif type(col_sample) == int: 
            N_cols = f[self.prefix]['axis0'].shape[0]
            col_sample_idx = np.random.choice(N_cols, size = col_sample, replace = False)
            col_sample_idx = sorted(list(col_sample_idx))
            columns = [col.decode() for col in f[self.prefix]['axis0'][col_sample_idx]]
        elif type(col_sample) == list and type(col_sample[0]) == int:
            col_sample_idx = sorted(col_sample)
            columns = [col.decode() for col in f[self.prefix]['axis0'][col_sample_idx]]
        elif type(col_sample) == list and type(col_sample[0]) == str:
            all_cols = np.array([col.decode() for col in f[self.prefix]['axis0'][:]])
            all_cols_sorted = np.sort(all_cols)
            all_cols_sorted_idx = np.argsort(all_cols)
            col_sample_pseudo_idx = np.searchsorted(all_cols_sorted, col_sample)
            col_sample_idx = all_cols_sorted_idx[col_sample_pseudo_idx]
            col_sample_idx = np.sort(col_sample_idx)
            columns = all_cols[col_sample_idx]
        
            
        out = f[self.prefix]['block0_values'][rows.index_col.values,:][:,col_sample_idx]
        
        out = pd.DataFrame(out, index = rows.cell_id, columns = columns)
        
        f.close()

        return out, row_sample_idx
