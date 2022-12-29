import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal, invwishart
from functools import wraps
from dask.distributed import Client, wait, progress

def post_hoc_permuation(y_true, y_score, n_permutations=10000, score_function=roc_auc_score, seed=None, n_jobs=None, verbose=False): 
    if seed:
        np.random.seed(seed)
    score = score_function(y_true, y_score)
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
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
        score, null, p = post_hoc_permuation(y_true[holdout_idx], y_pred[holdout_idx, 1], n_jobs=-1, verbose=True)
        all_score.append(score)
        all_null.append(null)
    score = np.mean(all_score)
    avg_null = np.vstack(all_null).mean(0)
    pvalue = (np.sum(np.mean(all_score) <= avg_null)+1.)/(all_null.shape[1]+1.)
    return score, avg_null, pvalue

def random_data_gen(n_samples=1000, n_feats=10, maha=1.0, ratio=0.5, seed=None):
    if seed:
        np.random.seed(seed)
    random_matrix = lambda n: np.dot(mat:=np.random.randn(n, n), mat.T)
    ## initialize multivariate normal dist with normally distributed means and covariance
    ## drawn from an inverse wishart distribution (conjugate prior for MVN)
    norm_means_a = np.random.randn(n_feats)
    norm_means_b = np.zeros_like(norm_means_a)
    wishart_cov = invwishart(n_feats, random_matrix(n_feats)).rvs()
    dist = mahalanobis(norm_means_a, norm_means_b, wishart_cov)
    norm_means_a = norm_means_a * (maha / dist)
    assert np.isclose(mahalanobis(norm_means_a, norm_means_b, wishart_cov), maha)
    ## multivariate normal distributions with different means and equal variances
    mvn_a = multivariate_normal(mean=norm_means_a, cov=wishart_cov)
    mvn_b = multivariate_normal(mean=norm_means_b, cov=wishart_cov)
    ## generate data samples from a multivariate normal
    data = np.vstack([mvn_a.rvs(int(n_samples*ratio)), mvn_b.rvs(n_samples - int(n_samples*ratio))])
    labels = np.arange(len(data))<int(n_samples*ratio)
    return data, labels

## decorator factory for simulation
def simulate(parameter_range, n_sim):
    """
    Decorator factory for simulating a function over a range of parameters. 
    """
    n_params = len(parameter_range)
    def decorator(function):
        wraps(function)
        print(f"Running {n_sim} simulations")
        try:
            client = Client.current()
            print(f"using dask client at {client.dashboard_link}")
            def wrapper(*args, **kwargs):
                print(f"Running {n_sim} simulations")
                print(f"Using dask client at {client.dashboard_link}")
                futures=[]
                for i in range(n_sim):
                    for p in parameter_range:
                        futures.append(client.submit(function, *args, param=p, seed=i, retries=1, **kwargs))
                print(f"{len(futures)} parallel jobs")
                # pbar = progress(futures, notebook=True)
                # display(pbar)
                wait(futures)
                gathered_futures = [f.result() if f.status=='finished' else None for f in futures]
                result = {p:{} for p in parameter_range}
                for i in range(len(futures)):
                    result[parameter_range[i%n_params]][i//n_params] = gathered_futures[i]
                return result

        except ValueError:
            print("No dask client avaialable, running sequentially")
            def wrapper(*args, **kwargs):
                print("No dask client avaialable, running sequentially")
                result = {p:{} for p in parameter_range}
                for i in range(n_sim):
                    for p in parameter_range:
                        result[p][i] = function(*args, param=p, **kwargs)
                return result
        return wrapper
    return decorator