import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal, invwishart
from functools import wraps, partial
from dask.distributed import Client, wait, progress

def post_hoc_permutation(y_true, y_score, n_permutations=10000, score_function=roc_auc_score, seed=None, n_jobs=None, backend="threading", verbose=False): 
    if seed:
        np.random.seed(seed)
    ## observed score
    score = score_function(y_true, y_score)
    ## run permutations in parallel to generate null scores
    permutation_scores = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(score_function)(
            np.random.choice(y_true, len(y_true), replace=False),
            y_score
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.) / (n_permutations + 1.)
    return score, permutation_scores, pvalue 

def post_hoc_permutation_cv(y_true, y_pred, cv):
    holdout_sets = [test for _, test in cv.split(y_true)]
    all_score = []
    all_null = []
    for holdout_idx in holdout_sets:
        score, null, p = post_hoc_permutation(y_true[holdout_idx], y_pred[holdout_idx, 1], n_jobs=-1, verbose=True)
        all_score.append(score)
        all_null.append(null)
    score = np.mean(all_score)
    avg_null = np.vstack(all_null).mean(0)
    pvalue = (np.sum(np.mean(all_score) <= avg_null)+1.)/(all_null.shape[1]+1.)
    return score, avg_null, pvalue

def random_data_gen(n_samples=1000, n_feats=10, maha=1.0, psi_diag=1.0, psi_offdiag=0., ddof=150, class_ratio=0.5, seed=None):
    if seed:
        np.random.seed(seed)
    ## initialize multivariate normal dist with normally distributed means and covariance
    ## drawn from an inverse wishart distribution (conjugate prior for MVN)
    norm_means_a = np.random.randn(n_feats)
    norm_means_b = np.zeros_like(norm_means_a)
    psi = psi_diag * np.eye(n_feats) + psi_offdiag * ~np.eye(n_feats).astype(bool)
    nu = n_feats + ddof
    wishart_cov = invwishart(nu, psi).rvs()
    ## specify the mahalanobis distance between the two distributions
    dist = mahalanobis(norm_means_a, norm_means_b, np.linalg.inv(wishart_cov))
    norm_means_a = norm_means_a * (maha / dist)
    assert np.isclose(mahalanobis(norm_means_a, norm_means_b, np.linalg.inv(wishart_cov)), maha)
    ## multivariate normal distributions with different means and equal variances
    mvn_a = multivariate_normal(mean=norm_means_a, cov=wishart_cov)
    mvn_b = multivariate_normal(mean=norm_means_b, cov=wishart_cov)
    ## not used, but compute correlations
    #corr = (D:=np.diag(1/np.sqrt(np.diag(wishart_cov)))) @ wishart_cov @ D
    ## generate data samples from a multivariate normal
    data = np.vstack([
        mvn_a.rvs(int(n_samples*class_ratio)),
        mvn_b.rvs(n_samples - int(n_samples*class_ratio))
    ])
    labels = np.arange(len(data))<int(n_samples*class_ratio)
    return data, labels

# def random_data_gen(n_samples=1000, n_feats=10, maha=1.0, ratio=0.5, seed=None):
#     if seed:
#         np.random.seed(seed)
#     ## initialize multivariate normal dist with normally distributed means and covariance
#     ## drawn from an inverse wishart distribution (conjugate prior for MVN)
#     norm_means_a = np.random.randn(n_feats)
#     norm_means_b = np.zeros_like(norm_means_a)
#     wishart_cov = invwishart(n_feats+1, np.identity(n_feats)).rvs()
#     dist = mahalanobis(norm_means_a, norm_means_b, wishart_cov)
#     norm_means_a = norm_means_a * (maha / dist)
#     assert np.isclose(mahalanobis(norm_means_a, norm_means_b, wishart_cov), maha)
#     ## multivariate normal distributions with different means and equal variances
#     corr = (D:=np.diag(1/np.sqrt(np.diag(wishart_cov)))) @ wishart_cov @ D
#     print("Correlation matrix:\n", corr)
#     mvn_a = multivariate_normal(mean=norm_means_a, cov=wishart_cov)
#     mvn_b = multivariate_normal(mean=norm_means_b, cov=wishart_cov)
#     ## generate data samples from a multivariate normal
#     data = np.vstack([mvn_a.rvs(int(n_samples*ratio)), mvn_b.rvs(n_samples - int(n_samples*ratio))])
#     labels = np.arange(len(data))<int(n_samples*ratio)
#     shuffle_idx = np.random.choice(np.arange(n_samples), n_samples, replace=False)
#     data, labels = data[shuffle_idx], labels[shuffle_idx]
#     return data, labels
