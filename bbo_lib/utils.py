import numpy as np
import pandas as pd
import os

''' Converts black box functions from np.arrays to pandas.DataFrames for easier interpretation of tabular data
Inputs:
    npa_x       inputs x1,x2,... as np array
    npa_y       output y as np array

Outputs:
    df          Dataframe with columns x1, x2, ... and y
'''

def fcn_as_df(npa_x,npa_y):
    # assure that even single rows have two dimensions - avoids breaking code below
    if npa_x.ndim == 1:
        npa_x = npa_x.reshape(1,-1)
        npa_y = npa_y.reshape(1,-1)

    n_col = npa_x.shape[1]
    col_names = [f"x{i+1}" for i in range(n_col)]
    df = pd.DataFrame(npa_x,columns=col_names)
    df["y"] = np.ravel(npa_y).T
    return df

''' Helper - Reads numpy arrays for each of the eight functions from disk and returns a list of dataframes containing the input and output info

Inputs: 
    path_to_fcns    Directory path - must contain folders "function_1","function_2"... each containing inputs.npy and outputs.npy 

Outputs: 
    fcn_dict - Dictionary of dataframes for each function, each dataframe containing columns x1, x2,..., y
'''


def read_fcns_from_disk(path_to_fcns):
    f_in = []
    f_out = []
    num_functions = 8
    # read the functions as a list of numpy arrays, separate for inputs x and output y
    for i in range(0, num_functions):
        dir_path = os.path.join(path_to_fcns, f'function_{i+1}')
        inputs = np.load(os.path.join(dir_path, 'inputs.npy'))
        outputs = np.load(os.path.join(dir_path, 'outputs.npy'))
        f_in.append(inputs)
        f_out.append(outputs)
        
    # create a dictionary of form {"f_1": DataFrame, "f_2": DataFrame, } where the DataFrames have column names x1, x2, ..., y
    fcn_dict = {}
    for i in range(0, num_functions):
        npa_x = f_in[i]
        npa_y = f_out[i]

        fcn_dict[f"f_{i+1}"] = fcn_as_df(npa_x,npa_y)

    return fcn_dict

# Helper function to print query point in expected format
def format_query(queryPoint):
    
    print(f"queryPoint={queryPoint}")

    # print the elements of the vector queryPoint, in the format 0.000000-0.000000-0.000000
    # where there is one 0.000000 per element in the vector

    query_str = '-'.join([f"{elem:.6f}" for elem in np.ravel(queryPoint)])
    print(f"queryPoint (formatted)={query_str}")

    #query_str=""
    #query_str.join(f"{queryPoint[0]:.6f}")
    #query_str.join([f"{elem:.6f}" for elem in np.ravel(queryPoint[1:])])
    #print(f"queryPoint (formatted)={query_str}")
