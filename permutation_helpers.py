import numpy as np
from joblib.parallel import Parallel, delayed
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.base import clone, BaseEstimator
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal, invwishart
from typing import Optional, Sequence, Tuple, Dict
from sklearn.model_selection import BaseCrossValidator


def score_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute performance metrics based on given predictions (output from predict_proba)
    and labels.
    Returns a dictionary with the following metrics:
    - roc_auc
    - accuracy
    - log_loss
    - brier_score
    """
    # predictions are 1 if the probability of the positive class is greater than 0.5
    y_pred_proba = np.array(y_pred)
    y_pred_disc = (y_pred_proba > 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred_disc)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba, pos_label=1)),
    }


def post_hoc_permutation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_permutations: int = 10000,
    score_function=roc_auc_score,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    backend: str = "threading",
    verbose: bool = False,
) -> Tuple[float, Sequence[float]]:
    """
    Permutes the labels and computes the score function for each permutation to generate a null distribution of scores.
    """
    if seed is not None:
        np.random.seed(seed)
    score = score_function(y_true, y_score)
    permutation_scores = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(score_function)(np.random.choice(y_true, len(y_true), replace=False), y_score)
        for _ in range(n_permutations)
    )
    return score, permutation_scores


def compute_p_value(score, null_scores):
    # assert isinstance(null_scores, Sequence)
    # assert isinstance(score, float)
    # assert null_scores, "Null scores cannot be empty."
    # assert not any(np.isnan(null_scores)), "Null scores cannot contain NaNs."
    pvalue = (np.sum(np.array(null_scores) >= score) + 1.0) / (len(null_scores) + 1.0)
    return pvalue


# def post_hoc_permutation_cv(
#     X: np.ndarray,
#     y: np.ndarray,
#     model: BaseEstimator,
#     cv: BaseCrossValidator,
#     n_permutations: int = 1000,
#     score_func=roc_auc_score,
#     n_jobs: Optional[int] = None,
#     verbose: bool = False,
# ) -> Tuple[float, np.ndarray, float]:
#     """
#     Perform post-hoc permutation tests using cross-validation.

#     Parameters
#     ----------
#     X : np.ndarray
#         Feature matrix.
#     y : np.ndarray
#         Target vector.
#     model : BaseEstimator
#         The machine learning model to train and evaluate.
#     cv : BaseCrossValidator
#         Cross-validation strategy to use.
#     n_permutations : int
#         Number of permutations to perform.
#     score_func : callable
#         Scoring function to evaluate the model.
#     n_jobs : int, optional
#         Number of jobs to run in parallel.
#     verbose : bool, optional
#         If True, print progress messages.

#     Returns
#     -------
#     score : float
#         Average score across all folds with original labels.
#     avg_null : np.ndarray
#         Average null scores across all folds.
#     pvalue : float
#         P-value based on the null distribution.
#     """
#     holdout_sets = [test for _, test in cv.split(y)]
#     all_score = []
#     all_null = []

#     for holdout_idx in holdout_sets:
#         # Split the data into training and holdout sets
#         X_train, X_test = X[~np.isin(np.arange(len(y)), holdout_idx)], X[holdout_idx]
#         y_train, y_test = y[~np.isin(np.arange(len(y)), holdout_idx)], y[holdout_idx]

#         # Train the model
#         model.fit(X_train, y_train)
#         y_pred = model.predict_proba(X_test)[:, 1]

#         # Perform post-hoc permutation
#         score, null_scores = post_hoc_permutation(
#             y_test,
#             y_pred,
#             n_permutations=n_permutations,
#             score_function=score_func,
#             n_jobs=n_jobs,
#             verbose=verbose,
#         )

#         all_score.append(score)
#         all_null.append(null_scores)

#     # Calculate average score and null distribution
#     avg_score = np.mean(all_score)
#     avg_null = np.mean(all_null, axis=0)
#     pvalue = compute_p_value(avg_score, avg_null)

#     return avg_score, avg_null, pvalue


# Assuming each dictionary in all_scores has the same keys
# def compute_avg_score(all_scores: List[Dict[str, float]]) -> Dict[str, float]:
#     keys = all_scores[0].keys()
#     avg_score = {key: np.mean([d[key] for d in all_scores]) for key in keys}
#     return avg_score


# # Compute avg_null as a list of dictionaries
# def compute_avg_null(all_nulls: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
#     keys = all_nulls[0][0].keys()  # Assuming all dictionaries have the same keys
#     avg_null = [
#         {key: np.mean([d[key] for d in inner_list]) for key in keys}
#         for inner_list in all_nulls
#     ]
#     return avg_null


def post_hoc_permutation_cv_nested(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseEstimator,
    param_grid: Dict[str, Sequence[float]],
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    n_permutations: int = 1000,
    score_func=score_model,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, float]:
    all_scores = []
    all_nulls = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # penalty parameter tuning on the inner fold
        grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring=score_func)
        # fit the model on everything but the outer-fold holdout set
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        # predict on the holdout set
        y_pred = best_model.predict_proba(X_test)[:, 1]
        # run post-hoc permutation
        score, null_scores = post_hoc_permutation(
            y_test,
            y_pred,
            n_permutations=n_permutations,
            score_function=score_func,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        all_scores.append(score)
        all_nulls.append(null_scores)

    # compute average score and null distribution across outer folds
    avg_score = np.mean(all_scores)
    avg_null = np.mean(all_nulls, axis=0)
    # compute p-value
    pvalue = compute_p_value(avg_score, avg_null)
    return avg_score, avg_null, pvalue


def _train_score(estimator, X_train, X_test, y_train, y_test, score_func, shuffle_train=False):
    if shuffle_train:
        indices = np.random.default_rng().permutation(len(y_train))
        y_train = y_train[indices]
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict_proba(X_test)[:, 1]
    return score_func(y_test, y_pred)


def pre_training_permutation(
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    n_permutations: int,
    score_func,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
) -> Tuple[float, Sequence[float]]:
    score = _train_score(
        clone(estimator),
        X_train,
        X_test,
        y_train,
        y_test,
        score_func,
        shuffle_train=False,
    )
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_train_score)(
            clone(estimator),
            X_train,
            X_test,
            y_train,
            y_test,
            score_func,
            shuffle_train=True,
        )
        for _ in range(n_permutations)
    )
    return score, permutation_scores


def pre_training_permutation_cv_nested(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseEstimator,
    param_grid: Dict[str, Sequence[float]],
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    n_permutations: int = 1000,
    score_func=score_model,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, float]:
    """
    Perform pre-training permutation tests using cross-validation.
    Permutes the training labels within each outer fold.
    """
    all_scores = []
    all_nulls = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring=score_func)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # perform pre-training permutation on outer fold using best model from inner fold
        score, null_scores = pre_training_permutation(
            best_model,
            X_train,
            X_test,
            y_train,
            y_test,
            n_permutations,
            score_func,
            n_jobs,
            verbose,
        )
        all_scores.append(score)
        all_nulls.append(null_scores)

    avg_score = np.mean(all_scores)
    avg_null = np.mean(all_nulls, axis=0)
    # pvalue = compute_p_value(avg_score, avg_null)
    return avg_score, avg_null


# also try the originally proposed pre-training CV, where labels
# are shuffled across folds
def pre_training_permutation_cv_acrossfolds(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseEstimator,
    param_grid: Dict[str, Sequence[float]],
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    n_permutations: int = 1000,
    score_func=score_model,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, float]:
    """
    Perform pre-training permutation tests using cross-validation.
    Permutes the training/holdout labels across folds.
    """
    score = nested_cv_vanilla(X, y, model, param_grid, outer_cv, inner_cv, score_func, shuffle_labels=False)
    null_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(nested_cv_vanilla)(X, y, model, param_grid, outer_cv, inner_cv, score_func, shuffle_labels=True)
        for _ in range(n_permutations)
    )
    # compute p-value
    pvalue = compute_p_value(score, null_scores)

    return score, null_scores, pvalue


def nested_cv_vanilla(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseEstimator,
    param_grid: Dict[str, Sequence[float]],
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    score_func,
    shuffle_labels: bool = False,
):
    if shuffle_labels:
        shuffled_idx = np.random.default_rng().permutation(len(y))
        y = y[shuffled_idx]

    holdout_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(model, param_grid, cv=inner_cv, scoring=score_func)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        score = _train_score(
            best_model,
            X_train,
            X_test,
            y_train,
            y_test,
            score_func,
            shuffle_train=False,
        )
        holdout_scores.append(score)
    return np.mean(holdout_scores)


def random_data_gen(
    n_samples: int = 1000,
    n_feats: int = 10,
    maha: float = 1.0,
    psi_diag: float = 1.0,
    psi_offdiag: float = 0.0,
    ddof: int = 150,
    class_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
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
    # corr = (D:=np.diag(1/np.sqrt(np.diag(wishart_cov)))) @ wishart_cov @ D
    ## generate data samples from a multivariate normal
    data = np.vstack(
        [
            mvn_a.rvs(int(n_samples * class_ratio)),
            mvn_b.rvs(n_samples - int(n_samples * class_ratio)),
        ]
    )
    labels = np.arange(len(data)) < int(n_samples * class_ratio)

    data, labels = shuffle(data, labels, random_state=seed)

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
