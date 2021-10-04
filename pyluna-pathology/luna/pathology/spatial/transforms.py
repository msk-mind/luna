""" Higher-level transformation functions """

import pandas as pd
import scipy.stats
import numpy as np
from pathlib import Path

from luna.pathology.spatial.stats import *

def generate_k_function_statistics(cell_paths, method_data, main_index=None):
    """
    Compute K-function spatial statistics on given cell-data

    Args:
        cell_paths (str or list[str]): paths to a single or multiple FOV regions
        method_data (dict): Configuration:
                "index": (str, optional) Column containting the patient/desired ID, if available (overrides main_index)
                "phenotype1" : {
                        "name" : (str) Column name to query
                        'value' : (str) Phenotype string to match (e.g. CD68)
                },
                "phenotype2" : {
                        "name" : (str) Column name to query
                        'value' : (str) Phenotype string to match (e.g. panCK)
                },
                "count" : (bool) Flag to compute counting stats.
                "radius" : (float) Radius cutoff
                "intensity" : (str, optional) Column containing intensity information 
                "distance" : (bool) Flag to compute intensity-distance stats.
    
    Returns:
        pd.DataFrame: spatial statistics aggregated over FOVs
    """

    if type(cell_paths)==str:
        cell_paths = [cell_paths]

    print (cell_paths)

    agg_k_data = {}

    pheno1_col    = method_data["phenotype1"]["name"]
    pheno1_val    = method_data["phenotype1"]["value"]
    pheno2_col    = method_data["phenotype2"]["name"]
    pheno2_val    = method_data["phenotype2"]["value"]
    index_col     = method_data.get("index", None)
    radius        = method_data["radius"]
    count         = method_data["count"]
    distance      = method_data["distance"]
    intensity_col = method_data.get("intensity", None)

    indices = set()

    for cell_path in cell_paths:

        if Path(cell_path).suffix == ".parquet":
            df = pd.read_parquet( cell_path ) 
        elif Path(cell_path).suffix == ".csv":
            df = pd.read_csv( cell_path ) 
        else:
            raise RuntimeError(f"Invalid input data type {cell_path}")

        # Look up the index for this slice
        if index_col:
            index = df[method_data['index']].iloc[0]
            indices.add(index)

        # Create the data arrays
        pheno1 = df[df[pheno1_col] == pheno1_val]
        pheno2 = df[df[pheno2_col] == pheno2_val]
        p1XY = np.array(pheno1[["Centroid X µm","Centroid Y µm"]])
        p2XY = np.array(pheno2[["Centroid X µm","Centroid Y µm"]])

        if intensity_col:
            I = np.array(pheno2[intensity_col])
        else:
            I = []
            if distance:
                raise RuntimeError("Can't compute intensity-distance function without intensity information")

        if p1XY.size == 0:
            print(f"WARNING: List of phenotype 1 cells ({pheno1_val}) is empty for {index}")
        if p2XY.size == 0:
            print(f"WARNING: List of phenotype 2 cells ({pheno2_val}) is empty for {index}")

        # Compute the K function
        print (f"Running... {cell_path}")

        fov_k_data = Kfunction(p1XY, p2XY, radius, ls=True, count=count, intensity=I, distance=distance)

        for key in fov_k_data:
            if key in agg_k_data:
                np.append(agg_k_data[key],fov_k_data[key])
            else:
                agg_k_data[key] = fov_k_data[key]

    data_out = {}

    for kfunct in agg_k_data.keys():
        arr = agg_k_data[kfunct]
        if len(arr)==0: arr=[0]
        data_out.update( 
            {
                f"For_{pheno1_val}_Find_{pheno2_val}_at{radius}_{kfunct}_{intensity_col}_mean": np.mean(arr),
                f"For_{pheno1_val}_Find_{pheno2_val}_at{radius}_{kfunct}_{intensity_col}_variance": np.var(arr),
                f"For_{pheno1_val}_Find_{pheno2_val}_at{radius}_{kfunct}_{intensity_col}_skew": scipy.stats.skew(arr),
                f"For_{pheno1_val}_Find_{pheno2_val}_at{radius}_{kfunct}_{intensity_col}_kurtosis": scipy.stats.kurtosis(arr)
            }
        )



    df_slice_out = pd.DataFrame(data_out, index=[0]).astype(np.float64)

    if main_index is None:
        if not len(indices)==1: 
            raise RuntimeError (f"Multiple cell maps with different indices! Found: {indices}")
        main_index = indices.pop()
    
    
    df_slice_out['main_index'] = main_index
    df_slice_out = df_slice_out.set_index('main_index')

    print (df_slice_out)

    return df_slice_out