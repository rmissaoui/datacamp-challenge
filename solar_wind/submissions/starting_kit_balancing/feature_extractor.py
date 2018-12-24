from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        model_lstm = False
        X_df_new = X_df.copy()

        # remove Range F 14 cause it's null
        X_df_new = X_df_new.drop(columns=['Range F 14'])

        # remove some useless features
        #X_df_new = remove_useless_features(X_df_new)

        # std
        #rolling_std_features = ['Beta', 'B','Bz', 'V', 'Range F 11','Range F 7']
        rolling_std_features = X_df_new.columns
        for feature in rolling_std_features:
            X_df_new = compute_rolling_std(X_df_new, feature, '2h')
            #X_df_new = compute_rolling_std(X_df_new, feature, '10h')

        # quantile25
        #rolling_quantile25_features = ['Beta']
        # for feature in rolling_quantile25_features:
        #    X_df_new = compute_rolling_quantile(X_df_new, feature, '2h', q=0.25)

        # quantile75
        #rolling_quantile75_features = ['Beta']
        # for feature in rolling_quantile75_features:
        #    X_df_new = compute_rolling_quantile(X_df_new, feature, '2h', q=0.75)

        # median
        #rolling_median_features = ['Beta','Bz']
        rolling_median_features = rolling_std_features
        for feature in rolling_median_features:
            X_df_new = compute_rolling_median(X_df_new, feature, '2h')
            #X_df_new = compute_rolling_median(X_df_new, feature, '10h')

        #X_df_new = keep_features(X_df_new)

        # feature extraction PCA
        #X_df_new = computer_pca(X_df_new)

        # skew
        features_not2deskew = [
            'Beta_2h_median', 'Beta_10h_median', 'Beta', 'Beta_2h_std', 'Beta_10h_std']
        #X_df_new = deskew_features(X_df_new[X_df_new.columns.difference(feautres_not2deskew)])

        # boxcox_features
        #X_df_new =  boxcox_features(X_df_new)

        if model_lstm:
            return X_df_new.values.reshape(X_df_new.shape[0], 1, X_df_new.shape[1])

        return X_df_new


def keep_features(X_df_new):
    columns_2keep = ['Beta_2h_median',
                     'Beta_10h_median',
                     'Beta',
                     'Beta_2h_std',
                     'Beta_10h_std',
                     'V',
                     'B',
                     'Bz_10h_median',
                     'RmsBob',
                     'B_10h_std',
                     'Range F 7_10h_std',
                     'Vth']
    return X_df_new[columns_2keep]


def remove_useless_features(X_df_new):
    # remove some useless features
    # Range F 14 is a null feature
    # Vx and V are highly opositely correlated so (corr = -1) we remove one of them
    # Range F 8, Range F 9 anf Range F 10 are highly correlated, we keep only one of them
    # Bx_rms, By_rms and Bz_rms are highly correlated, we keep only one of them
    # /!\ Range F 6,<->7 (not that highly correlated)
    # /!\ Range F 5,<->4 (not that highly correlated)

    columns_2remove = ['Range F 5', 'Range F 6', 'Np_nl', 'V', 'Vx', 'Range F 8',
                       'Range F 9', 'Bx_rms', 'By_rms', 'Range F 0',
                       'Range F 1', 'Range F 13', 'Range F 3', 'Range F 4', 'Pdyn',
                       'Bz_rms', 'Range F 11', 'Range F 12', 'Range F 13']

    #"Range F 14", "Pdyn", "Range F 10", "Range F 8", "By_rms", "Bx_rms", "V"

    return X_df_new.drop(columns=columns_2remove)


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


def compute_rolling_median(data, feature, time_window, center=False):
    """
    For a given dataframe, compute the median over
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
    name = '_'.join([feature, time_window, 'median'])
    data[name] = data[feature].rolling(time_window, center=center).median()
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data


def compute_rolling_quantile(data, feature, time_window, q=0.75, center=False):
    """
    For a given dataframe, compute the quantile over
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
    name = '_'.join([feature, time_window, 'quantile', str(int(q*100))])
    data[name] = data[feature].rolling(
        time_window, center=center).quantile(q, interpolation='midpoint')
    data[name] = data[name].ffill().bfill()
    data[name].astype(data[feature].dtype)
    return data


def compute_Beta(data):
    """
    Compute the evolution of the Beta for data.

    The function assume data already has ['Np','B','Vth'] features.
    """
    try:
        data['Beta'] = 1e6 * data['Vth'] * data['Vth'] * constants.m_p * data[
            'Np'] * 1e6 * constants.mu_0 / (1e-18 * data['B'] * data['B'])
    except KeyError:
        ValueError('Error computing Beta,B,Vth or Np'
                   ' might not be loaded in dataframe')
    return data
