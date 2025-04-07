#!/usr/bin/env python

import os
import time
import warnings
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from typing import Iterable
import traceback

# Dask imports
from dask.distributed import Client, wait
from dask_jobqueue import SGECluster
from concurrent.futures import Future

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Custom imports
from permutation_helpers import (
    post_hoc_permutation,
    score_model,
    pre_training_permutation,
    random_data_gen,
)
from simulate import simulate

warnings.simplefilter("ignore", FutureWarning)


# helpful functions for debugging and monitoring Dask jobs
def get_exceptions(futures: Iterable[Future], params: Iterable = None) -> pd.DataFrame:
    if params is None:
        params = range(len(futures))
    exceptions = []
    for i, (param, future) in enumerate(zip(params, futures)):
        if future.status == "error":
            exceptions.append(
                pd.Series(
                    {
                        "param": param,
                        "exception": repr(future.exception()),
                        "traceback_obj": future.traceback(),
                    }
                )
            )
    if not len(exceptions):
        raise Exception("None of the given futures resulted in exceptions")
    exceptions = pd.concat(exceptions, axis=1).T
    exceptions.set_index("param", inplace=True)
    return exceptions


def print_traceback(error_df, index):
    traceback.print_tb(error_df.loc[index, "traceback_obj"])


def setup_dask_cluster():
    cluster = SGECluster(
        cores=1,
        memory="2GB",
        processes=1,
        queue="short.q",
        job_extra=["-t 1-120"],
        log_directory=os.path.join(os.environ["HOME"], "logs/"),
        local_directory=os.path.join(os.environ["HOME"], "dask-worker-space/"),
        walltime="03:59:00",
        name="permutations-{$SGE_TASK_ID}",
    )
    client = Client(cluster)
    cluster.scale(n=100)
    return client


def set_params(maha, save=True):
    ### shared parameters
    class_params = {
        "C": np.logspace(np.log10(1e-4), np.log10(1e5), 8),
        "class_weight": "balanced",
    }
    permutation_params = {"n_permutations": 5000}
    sim_params = {"n_sim": 500}

    data_dir = os.path.join(os.environ["HOME"], "data")
    results_dir = os.path.join(data_dir, "sim_results", f"maha_{maha:.1f}")
    file_params = {"save": True, "results_dir": results_dir}
    data_gen_params = {
        "maha": maha,
        "psi_diag": 1.0,
        "psi_offdiag": 0.0,
        "ddof": 150,
        "n_samples": 1000,
        "n_feats": 10,
        "class_ratio": 0.5,
    }
    ## default parameters for simulations
    default_params = {
        "sim": deepcopy(sim_params),
        "data_gen": deepcopy(data_gen_params),
        "classif": deepcopy(class_params),
        "perm": deepcopy(permutation_params),
        "file": deepcopy(file_params),
    }

    ## set up parameters for specific simulations
    samplesize_params = deepcopy(default_params)
    samplesize_params["sim"]["parameter_range"] = np.logspace(2, 5, 5).astype(int)
    samplesize_params["data_gen"].pop("n_samples")

    nfeats_params = deepcopy(default_params)
    nfeats_params["sim"]["parameter_range"] = np.logspace(1, 10, 5, base=2).astype(int)
    nfeats_params["data_gen"].pop("n_feats")

    ratio_params = deepcopy(default_params)
    ratio_params["sim"]["parameter_range"] = np.logspace(
        np.log10(0.01), np.log10(0.5), 5
    )
    ratio_params["data_gen"].pop("class_ratio")

    testsize_params = deepcopy(default_params)
    testsize_params["sim"]["parameter_range"] = np.logspace(
        np.log10(0.01), np.log10(0.5), 5
    )
    if save:
        pickle.dump(
            samplesize_params,
            open(
                f"settings/samplesize_params_maha_{data_gen_params['maha']:.1f}.pkl",
                "wb",
            ),
        )
        pickle.dump(
            nfeats_params,
            open(
                f"settings/nfeats_params_maha_{data_gen_params['maha']:.1f}.pkl", "wb"
            ),
        )
        pickle.dump(
            ratio_params,
            open(f"settings/ratio_params_maha_{data_gen_params['maha']:.1f}.pkl", "wb"),
        )
        pickle.dump(
            testsize_params,
            open(
                f"settings/testsize_params_maha_{data_gen_params['maha']:.1f}.pkl", "wb"
            ),
        )
    return (
        samplesize_params,
        nfeats_params,
        ratio_params,
        testsize_params,
        default_params,
    )


# Define post-training simulation functions
def simulate_samplesize_post(param=None, seed=None, simno=None, settings=None):
    settings = deepcopy(settings)
    X, y = random_data_gen(n_samples=param, seed=seed, **settings["data_gen"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    X_val, y_val = random_data_gen(n_samples=1000, seed=None, **settings["data_gen"])
    max_AUC = 0
    best_estimator = None
    for C in settings["classif"].pop("C"):
        estimator = LogisticRegression(**settings["classif"], C=C)
        estimator.fit(X=X_train, y=y_train)
        y_pred = estimator.predict_proba(X_val)[:, 1]
        AUC = roc_auc_score(y_score=y_pred, y_true=y_val)
        if AUC >= max_AUC:
            max_AUC = AUC
            best_estimator = estimator
    y_pred = best_estimator.predict_proba(X_test)[:, 1]
    n_permutations = settings["perm"]["n_permutations"]
    score, permutation_scores = post_hoc_permutation(
        y_true=y_test,
        y_score=y_pred,
        n_permutations=n_permutations,
        score_function=score_model,
        n_jobs=-1,
    )
    if settings["file"]["save"]:
        pickle.dump(
            (score, permutation_scores),
            open(
                os.path.join(
                    settings["file"]["results_dir"],
                    f"post_samplesize_{param:.4f}_simno_{simno:05}.pkl",
                ),
                "wb",
            ),
        )
    return score, permutation_scores


def simulate_testsize_post(param=None, seed=None, simno=None, settings=None):
    settings = deepcopy(settings)
    X, y = random_data_gen(seed=seed, **settings["data_gen"])
    ## Split into train-test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=param, shuffle=True
    )
    ## Simulate validation set
    X_val, y_val = random_data_gen(seed=None, **settings["data_gen"])
    ## iterate over possible penalty params
    max_AUC = 0
    best_estimator = None
    for C in settings["classif"].pop("C"):
        estimator = LogisticRegression(**settings["classif"], C=C)
        estimator.fit(X=X_train, y=y_train)
        y_pred = estimator.predict_proba(X_val)[:, 1]
        AUC = roc_auc_score(y_score=y_pred, y_true=y_val)
        if AUC >= max_AUC:
            max_AUC = AUC
            best_estimator = estimator
    ## use model with tuned penalty
    y_pred = best_estimator.predict_proba(X_test)[:, 1]
    ## permutations
    n_permutations = settings["perm"]["n_permutations"]
    score, permutation_scores = post_hoc_permutation(
        y_true=y_test,
        y_score=y_pred,
        n_permutations=n_permutations,
        score_function=score_model,
        n_jobs=-1,
    )
    if settings["file"]["save"]:
        pickle.dump(
            (score, permutation_scores),
            open(
                os.path.join(
                    settings["file"]["results_dir"],
                    f"post_testsize_{param:.4f}_simno_{simno:05}.pkl",
                ),
                "wb",
            ),
        )
    return score, permutation_scores


def simulate_nfeats_post(param=None, seed=None, simno=None, settings=None):
    settings = deepcopy(settings)
    ## Simulate dataset
    X, y = random_data_gen(n_feats=param, seed=seed, **settings["data_gen"])
    ## Split into train-test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    ## Simulate validation set
    X_val, y_val = random_data_gen(n_feats=param, seed=None, **settings["data_gen"])
    ## iterate over possible penalty params
    max_AUC = 0
    best_estimator = None
    for C in settings["classif"].pop("C"):
        estimator = LogisticRegression(**settings["classif"], C=C)
        estimator.fit(X=X_train, y=y_train)
        y_pred = estimator.predict_proba(X_val)[:, 1]
        AUC = roc_auc_score(y_score=y_pred, y_true=y_val)
        if AUC >= max_AUC:
            max_AUC = AUC
            best_estimator = estimator
    ## use model with tuned penalty
    y_pred = best_estimator.predict_proba(X_test)[:, 1]
    ## permutations
    n_permutations = settings["perm"]["n_permutations"]
    score, permutation_scores = post_hoc_permutation(
        y_true=y_test,
        y_score=y_pred,
        n_permutations=n_permutations,
        score_function=score_model,
        n_jobs=-1,
    )
    if settings["file"]["save"]:
        pickle.dump(
            (score, permutation_scores),
            open(
                os.path.join(
                    settings["file"]["results_dir"],
                    f"post_nfeats_{param:.4f}_simno_{simno:05}.pkl",
                ),
                "wb",
            ),
        )
    return score, permutation_scores


def simulate_ratio_post(param=None, seed=None, simno=None, settings=None):
    settings = deepcopy(settings)
    ## Simulate dataset
    X, y = random_data_gen(class_ratio=param, seed=seed, **settings["data_gen"])
    ## Split into train-test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    ## Simulate validation set, same as original dataset
    X_val, y_val = random_data_gen(class_ratio=param, seed=None, **settings["data_gen"])
    ## iterate over possible penalty params
    max_AUC = 0
    best_estimator = None
    for C in settings["classif"].pop("C"):
        estimator = LogisticRegression(**settings["classif"], C=C)
        estimator.fit(X=X_train, y=y_train)
        y_pred = estimator.predict_proba(X_val)[:, 1]
        AUC = roc_auc_score(y_score=y_pred, y_true=y_val)
        if AUC >= max_AUC:
            max_AUC = AUC
            best_estimator = estimator
    ## use model with tuned penalty
    y_pred = best_estimator.predict_proba(X_test)[:, 1]
    ## permutations
    n_permutations = settings["perm"]["n_permutations"]
    score, permutation_scores = post_hoc_permutation(
        y_true=y_test,
        y_score=y_pred,
        n_permutations=n_permutations,
        score_function=score_model,
        n_jobs=-1,
    )
    # save with pickle
    if settings["file"]["save"]:
        pickle.dump(
            (score, permutation_scores),
            open(
                os.path.join(
                    settings["file"]["results_dir"],
                    f"post_ratio_{param:.4f}_simno_{simno:05}.pkl",
                ),
                "wb",
            ),
        )
    return score, permutation_scores


def run_simulation(maha_values):
    for maha in maha_values:
        data_dir = os.path.join(os.environ["HOME"], "data")
        results_dir = os.path.join(data_dir, "sim_results", f"maha_{maha:.1f}")
        os.makedirs(results_dir, exist_ok=True)

        samplesize_params, nfeats_params, ratio_params, testsize_params, _ = set_params(
            maha, save=True
        )

        # Use the simulate decorator with the defined functions
        sim_samplesize_post = simulate(**samplesize_params["sim"])(
            simulate_samplesize_post
        )
        sim_nfeats_post = simulate(**nfeats_params["sim"])(simulate_nfeats_post)
        sim_ratio_post = simulate(**ratio_params["sim"])(simulate_ratio_post)
        sim_testsize_post = simulate(**testsize_params["sim"])(simulate_testsize_post)

        # Run simulations
        futures_samplesize_post = sim_samplesize_post(settings=samplesize_params)
        futures_nfeats_post = sim_nfeats_post(settings=nfeats_params)
        futures_ratio_post = sim_ratio_post(settings=ratio_params)
        futures_testsize_post = sim_testsize_post(settings=testsize_params)

        wait(
            futures_samplesize_post
            + futures_nfeats_post
            + futures_ratio_post
            + futures_testsize_post
        )
        time.sleep(60)


if __name__ == "__main__":
    print("#" * 10, "\nSetting up Dask cluster\n", "#" * 10)
    client = setup_dask_cluster()
    print("#" * 10, f"\nDask cluster available at {client.dashboard_link}\n", "#" * 10)
    maha_values = np.linspace(0.0, 1.5, 5)
    run_simulation(maha_values)
    print("#" * 10, "\nShutting down Dask cluster\n", "#" * 10)
    client.shutdown()
