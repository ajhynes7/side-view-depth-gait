"""Calculate results of comparing Kinect and Zeno gait metrics."""

import os

import pandas as pd
from scipy.stats import spearmanr, pearsonr

import analysis.stats as st
import modules.pandas_funcs as pf

results_dir = os.path.join('data', 'results')
match_dir = os.path.join('data', 'matching')

df_k_raw = pd.read_csv(
    os.path.join(results_dir, 'kinect_gait_metrics.csv'), index_col=0)

df_z_raw = pd.read_csv(
    os.path.join(results_dir, 'zeno_gait_metrics.csv'), index_col=0)

df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))

# Drop rows where file has no match
df_match = df_match.dropna(axis=0)

df_match_zeno = pd.merge(df_match, df_z_raw, left_on='Zeno', right_index=True)

df_total = pd.merge(
    df_match_zeno,
    df_k_raw,
    left_on='Kinect',
    right_index=True,
    suffixes=('_z', '_k'))

# Take columns from total DataFrame to get Kinect and Zeno data
df_k = df_total.filter(like='_k')
df_z = df_total.filter(like='_z')

# Remove suffixes from column names
df_k = df_k.rename(columns=lambda x: str(x)[:-2])
df_z = df_z.rename(columns=lambda x: str(x)[:-2])

# Group by gait metric without suffix
df_k_grouped = df_k.groupby(lambda x: x[:-2], axis=1).mean()
df_z_grouped = df_z.groupby(lambda x: x[:-2], axis=1).mean()

# Calculate results

funcs = {
    'pearson': lambda a, b: pearsonr(a, b)[0],
    'spearman': lambda a, b: spearmanr(a, b)[0],
    'abs_rel_error':
    lambda a, b: st.relative_error(a, b, absolute=True).mean(),
    'bias': lambda a, b: st.BlandAltman(a, b).bias,
    'range': lambda a, b: st.BlandAltman(a, b).range
}

dict_results_LR = {
    name: pf.apply_to_columns(df_k, df_z, func)
    for name, func in funcs.items()
}
dict_results_grouped = {
    name: pf.apply_to_columns(df_k_grouped, df_z_grouped, func)
    for name, func in funcs.items()
}

df_results_LR = pd.DataFrame(dict_results_LR).T
df_results_grouped = pd.DataFrame(dict_results_grouped).T

df_results_LR.to_csv(os.path.join('results', 'results_LR.csv'))
df_results_grouped.to_csv(os.path.join('results', 'results_grouped.csv'))
