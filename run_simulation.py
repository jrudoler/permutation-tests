from permutation_helpers import *
import warnings
warnings.simplefilter('ignore', FutureWarning)
import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (permutation_test_score, learning_curve, LeaveOneGroupOut,
                                     KFold, cross_val_score, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.utils import parallel_backend
import cmldask.CMLDask as da
from dask.distributed import Client


if __name__=="__main__":
    rhino_client = da.new_dask_client(
        job_name="cli_simulations",
        memory_per_job="1.5GB",
        max_n_jobs=400, threads_per_job=5, 
        adapt=True,
        local_directory="/home1/jrudoler/dask_worker_space",
        log_directory="/home1/jrudoler/logs/",
    )
       
    @simulate(parameter_range=np.linspace(0., 1.5, 10), n_sim=500, client=rhino_client)
    def simulate_maha_pre(param=None, seed=None):
        X, y = random_data_gen(n_samples=1000, n_feats=10, maha=param, ratio=0.5, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)
        estimator = LogisticRegressionCV(class_weight='balanced', Cs=4)
        n_permutations = 5000
        score, null, p = pre_training_permutation(
            estimator,
            X_train, X_test, y_train, y_test,
            n_permutations=n_permutations,
            score_func=roc_auc_score,
            verbose=True, n_jobs=-1
        )
        return score, null, p
    result, futures = simulate_maha_pre()
    df_result = pd.DataFrame(result).melt(var_name="param")
    df_result[["score", "perm_scores", "pval"]] = df_result['value'].apply(pd.Series)
    df_result = df_result.drop(columns='value')
    df_result.to_pickle("simulate_maha_pre.pkl")

    rhino_client.cancel(futures)
    rhino_client.shutdown()
    del rhino_client
    print("Done!")
