
import scipy.stats
import numpy as np

def compute_stats_1d(vec, fx_name_prefix, n_percentiles=4):
    """ Computes 1d (histogram)-like summary statistics
    
    Args:
        vec (np.array): a 1-d vector input
        fx_name_prefix (str): Prefix for feature names
        n_percentiles (int): Number of percentiles to compute, default 4 = 0 (min), 25, 50, 75, 100 (max)
    
    Returns:
        dict: summary statistics
    """
    n, _, sm, sv, ss, sk = scipy.stats.describe(vec)
    # ln_params = scipy.stats.lognorm.fit(vec, floc=0)

    hist_features = {
        f'{fx_name_prefix}_nobs': n,
        f'{fx_name_prefix}_mean': sm,
        f'{fx_name_prefix}_variance': sv,
        f'{fx_name_prefix}_skewness': ss,
        f'{fx_name_prefix}_kurtosis': sk,
        # f'{fx_name_prefix}_lognorm_fit_p0': ln_params[0],
        # f'{fx_name_prefix}_lognorm_fit_p2': ln_params[2]
    }
    
    percentiles = np.linspace(0, 100, n_percentiles + 1)

    for percentile, value in zip(percentiles, np.percentile(vec, percentiles)):
        hist_features[f'{fx_name_prefix}_pct{int(percentile)}'] = value

    return hist_features