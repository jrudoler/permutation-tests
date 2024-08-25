#!/home/stat/jrudoler/.cache/pypoetry/virtualenvs/permutation-tests-IPEyRD8n-py3.12/bin/python
print("Running aggregate_results.py")

print("Importing libraries")
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 14
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.transparent"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.1
plt.rcParams["axes.labelsize"] = 16

import seaborn as sns

sns.set_palette("deep")
import json

wharton_colors = json.load(open("wharton-colors-distinct.json", "r"))
wharton_pal = sns.color_palette(wharton_colors.values())
sns.set_palette(wharton_pal)
# set matplotlib color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=wharton_pal)

import numpy as np
import pandas as pd
import os
import pickle

from permutation_helpers import random_data_gen

from scipy.stats import norm


def auc_from_maha(maha_dist: float):
    # source: https://en.wikipedia.org/wiki/Sensitivity_index#RMS_sd_discriminability_index
    auc = norm.cdf(maha_dist / np.sqrt(2))
    return auc


maha_values = np.linspace(0.0, 1.5, 5)

# print("#### Testsize ####")
# testsize = None
# for m in maha_values.round(2):
#     print(f"maha={m}")
#     settings = pickle.load(open(f"settings/testsize_params_maha_{m:.1f}.pkl", "rb"))
#     nsim = settings["sim"]["n_sim"]
#     results_dir = settings["file"]["results_dir"]
#     param_range = settings["sim"]["parameter_range"]
#     for p in param_range:
#         for i in range(nsim):
#             try:
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"post_testsize_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 testsize_post = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"pre_testsize_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 testsize_pre = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 testsize_pre["test"] = "pre"
#                 testsize_post["test"] = "post"
#                 testsize_pre["d"] = m
#                 testsize_post["d"] = m
#                 testsize_pre["param"] = p
#                 testsize_post["param"] = p
#                 testsize = pd.concat([testsize, testsize_pre, testsize_post])
#             except FileNotFoundError:  # some errors when only one class is present
#                 print(f"File not found: param={p}, simno={i}, maha={m}")
# testsize = testsize.reset_index(names="simno")
# testsize.to_pickle("testsize.pkl")

# del testsize

# print("#### Samplesize ####")
# samplesize = None
# for m in maha_values.round(2):
#     print(f"maha={m}")
#     settings = pickle.load(open(f"settings/samplesize_params_maha_{m:.1f}.pkl", "rb"))
#     nsim = settings["sim"]["n_sim"]
#     results_dir = settings["file"]["results_dir"]
#     param_range = settings["sim"]["parameter_range"]
#     for p in param_range:
#         for i in range(nsim):
#             try:
#                 score, null = pd.read_pickle(
#                     os.path.join(
#                         results_dir, f"post_samplesize_{p:.4f}_simno_{i:05}.pkl"
#                     )
#                 )
#                 samplesize_post = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 score, null = pd.read_pickle(
#                     os.path.join(
#                         results_dir, f"pre_samplesize_{p:.4f}_simno_{i:05}.pkl"
#                     )
#                 )
#                 samplesize_pre = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 samplesize_pre["test"] = "pre"
#                 samplesize_post["test"] = "post"
#                 samplesize_pre["d"] = m
#                 samplesize_post["d"] = m
#                 samplesize_pre["param"] = p
#                 samplesize_post["param"] = p
#                 samplesize = pd.concat([samplesize, samplesize_pre, samplesize_post])
#             except FileNotFoundError:  # some errors when only one class is present
#                 print(f"File not found: param={p}, simno={i}, maha={m}")
# samplesize = samplesize.reset_index(names="simno")
# samplesize.to_pickle("samplesize.pkl")

# del samplesize

# print("#### NFeatures ####")
# nfeats = None
# for m in maha_values.round(2):
#     print(f"maha={m}")
#     settings = pickle.load(open(f"settings/nfeats_params_maha_{m:.1f}.pkl", "rb"))
#     nsim = settings["sim"]["n_sim"]
#     results_dir = settings["file"]["results_dir"]
#     param_range = settings["sim"]["parameter_range"]
#     for p in param_range:
#         for i in range(nsim):
#             try:
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"post_nfeats_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 nfeats_post = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"pre_nfeats_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 nfeats_pre = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 nfeats_pre["test"] = "pre"
#                 nfeats_post["test"] = "post"
#                 nfeats_pre["d"] = m
#                 nfeats_post["d"] = m
#                 nfeats_pre["param"] = p
#                 nfeats_post["param"] = p
#                 nfeats = pd.concat([nfeats, nfeats_pre, nfeats_post])
#             except FileNotFoundError:  # some errors when only one class is present
#                 print(f"File not found: param={p}, simno={i}, maha={m}")
# nfeats = nfeats.reset_index(names="simno")
# nfeats.to_pickle("nfeats.pkl")

# del nfeats

# print("#### Ratio ####")
# ratio = None
# for m in maha_values.round(2):
#     print(f"maha={m}")
#     settings = pickle.load(open(f"settings/ratio_params_maha_{m:.1f}.pkl", "rb"))
#     nsim = settings["sim"]["n_sim"]
#     results_dir = settings["file"]["results_dir"]
#     param_range = settings["sim"]["parameter_range"]
#     for p in param_range:
#         for i in range(nsim):
#             try:
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"post_ratio_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 ratio_post = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 score, null = pd.read_pickle(
#                     os.path.join(results_dir, f"pre_ratio_{p:.4f}_simno_{i:05}.pkl")
#                 )
#                 ratio_pre = pd.merge(
#                     pd.DataFrame(score, index=[i]),
#                     pd.DataFrame(null, i + np.zeros(len(null)).astype(int)).add_prefix(
#                         "null_"
#                     ),
#                     left_index=True,
#                     right_index=True,
#                 )
#                 ratio_pre["test"] = "pre"
#                 ratio_post["test"] = "post"
#                 ratio_pre["d"] = m
#                 ratio_post["d"] = m
#                 ratio_pre["param"] = p
#                 ratio_post["param"] = p
#                 ratio = pd.concat([ratio, ratio_pre, ratio_post])
#             except FileNotFoundError:  # some errors when only one class is present
#                 print(f"File not found: param={p}, simno={i}, maha={m}")
# ratio = ratio.reset_index(names="simno")
# ratio.to_pickle("ratio.pkl")

for parameter in ["testsize", "samplesize", "nfeats", "ratio"]:
    print(f"Running {parameter}")
    long_data = pd.read_pickle(f"{parameter}.pkl")
    long_data["param"] = long_data["param"].round(3)
    for metric in ["roc_auc", "accuracy", "brier_score", "log_loss"]:
        print(f"Evaluating {metric}")
        if metric in ["brier_score", "log_loss"]:
            # lower is better
            long_data["null_exceeds_score"] = (
                long_data[f"null_{metric}"] <= long_data[metric]
            ).astype(int)
        else:
            # higher is better
            long_data["null_exceeds_score"] = (
                long_data[f"null_{metric}"] >= long_data[metric]
            ).astype(int)

        agg_data = (
            long_data.groupby(["d", "param", "test", "simno", metric])
            .agg({"null_exceeds_score": lambda x: (x.sum() + 1) / (len(x) + 1)})
            .reset_index()
            .rename(columns={"null_exceeds_score": "pval"})
        )
        agg_data["positive"] = agg_data["pval"] <= 0.05
        print("Saving")
        agg_data.to_pickle(f"{parameter}_pval_{metric}.pkl")
