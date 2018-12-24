import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
import catboost as cb
from scipy.stats import mode
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

class Classifier(BaseEstimator):
    def __init__(self):
        cv_params = {'max_depth': [5, 6, 7],
                     'bagging_temperature': [1, 2],
                     'learning_rate': [0.13,0.11, 0.09, 0.01]}
        #cv_params = {'bagging_temperature': [2], 'learning_rate': [0.05], 'max_depth': [7]}
        ind_params = {'iterations': 150, 
                      'rsm': 0.75, 'random_state': 123,
                     'od_type':'Iter',
                     'od_wait':20,
                     'combinations_ctr':None,
                     'eval_metric':'BalancedAccuracy',
                     'l2_leaf_reg':40,
                      'verbose':50,
                     'gpu_ram_part':0.95}
        
        optimized_model = GridSearchCV(cb.CatBoostClassifier(**ind_params),
                                     cv_params,
                                     scoring = 'balanced_accuracy',
                                     cv = 5,
                                     n_jobs = -1)
        self.model = optimized_model

    def fit(self, X, y):
        self.models_ = []

        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            # Create data for this fold
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
            X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]

            # Run models for this fold
            self.models_.append(
                self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
            )


    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], 2))

        for model in self.models_:
            y_pred += model.predict_proba(X) / 5.

        s_y_pred = pd.Series(y_pred[:, 1])
        s_y_pred_smoothed = s_y_pred.rolling(65, min_periods=0, center=True).quantile(0.65)
        return np.array(list(zip(1.-s_y_pred_smoothed[:],s_y_pred_smoothed[:])))
    
##############################@
from __future__ import division
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
class Classifier(BaseEstimator):
    def __init__(self):
        cv_params = {'max_depth': [5, 7, 9], 'num_leaves': [5, 8, 11], 'learning_rate': [0.05, 0.09, 0.11]}
        ind_params = {'n_estimators': 100, 'subsample': 0.5, 'colsample_bytree': 0.6, 'objective': 'binary'}
        optimized_GBM = GridSearchCV(LGBMClassifier(**ind_params), cv_params, scoring = 'balanced_accuracy', cv = 5, n_jobs = -1)
        self.model = optimized_GBM
    def fit(self, X, y):
        print("\nentered fitting ...")
        self.model.fit(X, y)
        print("finished fitting ...")
    def predict_proba(self, X):
        yres = self.model.predict_proba(X)
        s_y_pred = pd.Series(yres[:, 1])
        s_y_pred_smoothed = s_y_pred.rolling(80, min_periods=0, center=True).quantile(0.6)
        return np.array(list(zip(1-s_y_pred_smoothed[:],s_y_pred_smoothed[:])))
