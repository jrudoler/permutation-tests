#!/usr/bin/env python

import os
from dask.distributed import Client
from dask_jobqueue import SGECluster
from simulate import simulate
from permutation_helpers import (
    post_hoc_permutation_cv_nested,
    random_data_gen,
    score_model,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import numpy as np
from copy import deepcopy
import pickle


def set_params(maha, save=True):
    ### shared parameters
    class_params = {
        "C": np.logspace(np.log10(1e-4), np.log10(1e5), 8),
        "class_weight": "balanced",
    }
    permutation_params = {"n_permutations": 5000}
    sim_params = {
        "n_sim": 500,
    }

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
    ## use default parameters and update the ones that need to be changed
    ## for each simulation, we will vary one parameter at a time and remove
    ## the corresponding key from the data_gen dictionary to avoid conflicts

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


# for maha in maha_values:
#     data_dir = os.path.join(os.environ["HOME"], "data")
#     results_dir = os.path.join(data_dir, "sim_results", f"maha_{maha:.1f}")
#     ## set up directories for saving results
#     ## results are separated by the the underlying probability distributions (and their mahalanobis distance)
#     os.makedirs(results_dir, exist_ok=True)


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


def simulate_samplesize_cv_nested(param=None, seed=None, simno=None, settings=None):
    settings = deepcopy(settings)
    X, y = random_data_gen(n_samples=param, seed=seed, **settings["data_gen"])

    inner_cv = KFold(n_splits=5)
    outer_cv = KFold(n_splits=5)

    score, null_scores, pvalue = post_hoc_permutation_cv_nested(
        X=X,
        y=y,
        model=LogisticRegression(),
        param_grid=settings["classif"],
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        score_func=score_model,
    )
    if settings["file"]["save"]:
        os.makedirs(settings["file"]["results_dir"], exist_ok=True)
        file_path = os.path.join(
            settings["file"]["results_dir"],
            f"cv_post_samplesize_{param:.4f}_simno_{simno:05}.pkl",
        )
        pickle.dump(
            {"score": score, "null_scores": null_scores, "pvalue": pvalue},
            open(file_path, "wb"),
        )


def run_simulation(maha):
    # Define simulation parameters
    samplesize_params, nfeats_params, ratio_params, testsize_params, _ = set_params(
        maha, save=False
    )

    sim_samplesize = simulate(**samplesize_params["sim"])(simulate_samplesize_cv_nested)

    sim_samplesize(settings=samplesize_params)


if __name__ == "__main__":
    # Set up Dask cluster
    client = setup_dask_cluster()

    maha_values = np.linspace(0.0, 1.5, 5)
    for maha in maha_values:
        run_simulation(maha)
    client.shutdown()
