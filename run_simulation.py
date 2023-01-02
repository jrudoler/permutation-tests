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

@simulate(parameter_range=np.linspace(0., 1.5, 25), n_sim=1000)
def simulate_maha(param=None, seed=None):
    X, y = random_data_gen(n_samples=5000, n_feats=5, maha=param, ratio=0.5, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)
    estimator = LogisticRegressionCV(class_weight='balanced', Cs=4)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    n_permutations = 5000
    estimator.fit(X=X_train, y=y_train)
    y_pred = estimator.predict_proba(X_test)[:, 1]
    score, permutation_scores, pvalue = post_hoc_permuation(
        y_true=y_test, y_score=y_pred,
        n_permutations=n_permutations, n_jobs=-1,
        )
    return score, permutation_scores, pvalue

if __name__=="__main__":
    rhino_client = da.new_dask_client(
        job_name="simulations",
        memory_per_job="1.5GB",
        max_n_jobs=400, threads_per_job=5, 
        adapt=False,
        local_directory="/home1/jrudoler/dask_worker_space",
        log_directory="/home1/jrudoler/logs/",
    )
    rhino_client.cluster.scale(400)
    with rhino_client.as_current():
        results, futures = simulate_maha()
    df_result = pd.DataFrame(results).melt(var_name="param")
    df_result[["score", "perm_scores", "pval"]] = df_result['value'].apply(pd.Series)
    df_result = df_result.drop(columns='value')
    df_result.to_pickle("simulate_samplesize.pkl")
    print("Done!")
