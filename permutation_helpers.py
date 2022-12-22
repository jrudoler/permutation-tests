import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.metrics import roc_auc_score

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

def post_hoc_permutation_cv(y_true, y_pred):
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