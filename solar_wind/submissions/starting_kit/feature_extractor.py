from scipy import constants
import numpy as np
from itertools import product
 
class FeatureExtractor(object):
    def __init__(self):
        pass
    def fit(self, X_df, y):
        return self
    def transform(self, X_df):
        X_df_new = X_df.copy()
                # Beta
        X_df_new = compute_rolling_min(X_df_new, 'Beta', '20h')
        X_df_new = compute_rolling_min(X_df_new, 'Beta', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'Beta', '2d')
        X_df_new = compute_rolling_min(X_df_new, 'Beta', '35h')
        X_df_new = compute_rolling_max(X_df_new, 'Beta', '1d')
        X_df_new = compute_rolling_median(X_df_new, 'Beta', '2h')
        
    
        # B
        X_df_new = compute_rolling_min(X_df_new, 'B', '20h')
        X_df_new = compute_rolling_min(X_df_new, 'B', '2h')
        X_df_new = compute_rolling_max(X_df_new, 'B', '20h')
        
        # By_rms
        X_df_new = compute_rolling_median(X_df_new, 'By_rms', '2h')
        X_df_new = compute_rolling_median(X_df_new, 'By_rms', '20h')
 
 
        # Bx_rms
        X_df_new = compute_rolling_min(X_df_new, 'Bx_rms', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'Bx_rms', '20h')
 
 
        # Bz_rms
        X_df_new = compute_rolling_min(X_df_new, 'Bz_rms', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'Bz_rms', '20h')
        X_df_new = compute_rolling_median(X_df_new, 'Bz_rms', '2h')
        
        # V
        # X_df_new = compute_rolling_std(X_df_new, 'V', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'V', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'V', '20h')
        
        # RmsBob
        X_df_new = compute_rolling_min(X_df_new, 'RmsBob', '20h')
        X_df_new = compute_rolling_median(X_df_new, 'RmsBob', '2h')
        
        # Vth
        X_df_new = compute_rolling_min(X_df_new, 'Vth', '2h')
        X_df_new = compute_rolling_min(X_df_new, 'Vth', '20h')
 
        
        X_df_new = compute_rolling_std(X_df_new, 'Beta', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'B', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'Vth', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V', '6h')       
        X_df_new = compute_rolling_std(X_df_new, 'RmsBob', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'Beta', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'Beta', '2d')
        X_df_new = compute_rolling_std(X_df_new, 'B', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'Vth', '18h')
        X_df_new = compute_rolling_std(X_df_new, 'V', '18h')       
        X_df_new = compute_rolling_std(X_df_new, 'RmsBob', '18h')
        
        X_df_new = cart_to_sph(X_df_new, 'B', 'Bx', 'Bz')
        X_df_new = cart_to_sph(X_df_new, 'V', 'Vx', 'Vz')
        
        X_df_new = compute_rolling_std(X_df_new, 'B_phi', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'B_theta', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V_phi', '6h')
        X_df_new = compute_rolling_std(X_df_new, 'V_theta', '6h')
        X_df_new = compute_rolling_quantile(X_df_new, 'Beta', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'B', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'Vth', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'V', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'RmsBob', 9, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'Beta', 20, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'B', 20, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'Vth', 20, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'V', 20, center=True)
        X_df_new = compute_rolling_quantile(X_df_new, 'RmsBob', 20, center=True)
        X_df_new = compute_rolling_mean(X_df_new, 'Beta', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'B', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'Vth', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'V', '6h')       
        X_df_new = compute_rolling_mean(X_df_new, 'RmsBob', '6h')
        X_df_new = compute_rolling_mean(X_df_new, 'Beta', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'B', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'Vth', '18h')
        X_df_new = compute_rolling_mean(X_df_new, 'V', '18h')       
        X_df_new = compute_rolling_mean(X_df_new, 'RmsBob', '18h')
        
        X_df_new = compute_rolling_skew(X_df_new, 'Beta', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'B', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'Vth', 60, center=True)
        X_df_new = compute_rolling_skew(X_df_new, 'V', 60, center=True)       
        X_df_new = compute_rolling_skew(X_df_new, 'RmsBob', 60, center=True)
        
        X_df_new = compute_rolling_kurt(X_df_new, 'Beta', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'B', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'Vth', '60h', center=True)
        X_df_new = compute_rolling_kurt(X_df_new, 'V', '60h', center=True)       
        X_df_new = compute_rolling_kurt(X_df_new, 'RmsBob', '60h', center=True)
        
        #data_polynomial = X_df.copy()
        #data_polynomial = compute_polynomial_features(data_polynomial, ['Beta','B','RmsBob'])
        #X_df_new = pd.concat([X_df_new, data_polynomial], axis=1)
 
        columns_ext = ['V_phi', 'V_theta', 'B_phi', 'B_theta',
                        'Beta', 'Vth', 'B', 'V', 'RmsBob']
        X_df_new = col_ext(X_df_new, columns_ext)
        
        features2keep = ['Beta_18h_mean',
                 'Beta_1d_max',
                 'B_phi_6h_std',
                 'Vth_2h_min',
                 'Vth_6h_mean',
                 'B_6h_mean',
                 'B_2h_min',
                 'B_18h_mean',
                 'Vth_18h_std',
                 'Beta_6h_std',
                 'V_18h_mean',
                 'B_20h_min',
                 'B_theta_6h_std',
                 'Vth_60h_kurt',
                 'Vth_20h_min',
                 'Vth_9_quantile',
                 'V_2h_min',
                 'RmsBob_18h_mean',
                 'RmsBob_2h_median',
                 'RmsBob_6h_std',
                 'B_18h_std',
                 'RmsBob_20h_min',
                 'Beta_35h_min',
                 'B_20h_max',
                 'RmsBob_6h_mean',
                 'Beta_2h_min',
                 'Beta_2d_min',
                 'RmsBob_9_quantile',
                 'Beta_6h_mean',
                 'Beta_20h_min',
                 'V_20h_min',
                 'Beta_2h_median',
                 'Beta_9_quantile']
        X_df_new = X_df_new[features2keep]
        for feature in X_df_new.columns:
            X_df_new = compute_rolling_quantile(X_df_new, feature, 9, center=True)
            X_df_new = compute_rolling_quantile(X_df_new, feature, 20, center=True)
            X_df_new = compute_rolling_median(X_df_new, feature, '2h')
        
        features2keep2 = ['Beta_2h_min_2h_median',
             'Beta_35h_min_2h_median',
             'RmsBob_9_quantile_2h_median',
             'Vth_9_quantile',
             'RmsBob_9_quantile_9_quantile',
             'V_20h_min_2h_median',
             'RmsBob_6h_mean_2h_median',
             'Beta_35h_min_20_quantile',
             'Beta_6h_mean_20_quantile',
             'Beta_2h_min_9_quantile',
             'Beta_35h_min_9_quantile',
             'RmsBob_6h_mean',
             'Beta_6h_mean',
             'Vth_60h_kurt',
             'Beta_20h_min_2h_median',
             'Beta_20h_min_20_quantile',
             'Beta_6h_mean_2h_median',
             'RmsBob_2h_median_20_quantile',
             'V_20h_min_20_quantile',
             'Beta_20h_min_9_quantile',
             'Beta_2d_min_20_quantile',
             'Beta_2d_min_9_quantile',
             'Vth_18h_std_2h_median',
             'RmsBob_6h_mean_9_quantile',
             'B_18h_std_2h_median',
             'V_2h_min_2h_median',
             'Beta_6h_mean_9_quantile',
             'Beta_2h_median_2h_median',
             'Beta_9_quantile_2h_median',
             'RmsBob_6h_std_2h_median',
             'B_18h_std_20_quantile',
             'Beta_2h_median_9_quantile',
             'RmsBob_20h_min',
             'B_20h_max_2h_median',
             'RmsBob_6h_mean_20_quantile',
             'V_20h_min_9_quantile',
             'Beta_35h_min',
             'Beta_2d_min',
             'RmsBob_9_quantile',
             'Beta_2h_median',
             'Beta_2h_min',
             'B_20h_max_20_quantile',
             'Beta_9_quantile_9_quantile',
             'V_20h_min',
             'Beta_2h_min_20_quantile',
             'RmsBob_9_quantile_20_quantile',
             'Beta_20h_min',
             'Beta_9_quantile_20_quantile',
             'Beta_2h_median_20_quantile',
             'Beta_9_quantile']
        #X_df_new = X_df_new[features2keep2]
            
        return X_df_new

################
def compute_polynomial_features(data, features):
 
    data_new = data.copy()
 
    for i, j in product(features, repeat=2):
        name = i + '_inter_' + j
        data_new[name] = data_new[i].values * data_new[j].values
        data_new[name].astype(data_new[i].dtype)
 
    return data_new

################
def col_ext(data, columns_ext):
    return data.loc[:, [xx for xx in data.columns if xx not in columns_ext]]
 
def cart_to_sph(data, feature, featureX, featureZ):
    namePhi = '_'.join([feature, 'phi'])
    nameTheta = '_'.join([feature, 'theta']) 
    data[nameTheta] = np.arccos(data[featureZ]/data[feature])
    data[namePhi] = np.arccos(data[featureX]/(data[feature]*np.sin(data[nameTheta])))
    data[nameTheta] = data[nameTheta].ffill().bfill()
    data[namePhi] = data[namePhi].ffill().bfill()
    return data
 
def compute_rolling_skew(data, feature, time_window, center=False):
    name = '_'.join([feature, str(time_window), 'skew'])
    data[name] = data[feature].rolling(time_window, center=center).skew()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_kurt(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'kurt'])
    data[name] = data[feature].rolling(time_window).kurt()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_mean(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'mean'])
    data[name] = data[feature].rolling(time_window, center=center).mean()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_quantile(data, feature, time_window, center=False):
    name = '_'.join([feature, str(time_window), 'quantile'])
    data[name] = data[feature].rolling(time_window, center=center).quantile(0.75)
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_std(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature
 
    Parameters
    ----------
    data : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling mean from
    time_indow : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    data[name] = data[feature].rolling(time_window, center=center).std()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_min(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'min'])
    data[name] = data[feature].rolling(time_window, center=center).min()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_max(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'max'])
    data[name] = data[feature].rolling(time_window, center=center).max()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data
 
def compute_rolling_median(data, feature, time_window, center=False):
    name = '_'.join([feature, time_window, 'median'])
    data[name] = data[feature].rolling(time_window, center=center).median()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data