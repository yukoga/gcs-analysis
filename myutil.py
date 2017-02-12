from collections import deque
from itertools import cycle
from sklearn.preprocessing import minmax_scale, scale
import numpy as np
import pandas as pd


def gen_multi_normal_data(n_samples=1, n_features=2,
                          mean=None, var=None,
                          v_bound=3, shift=0, do_abs=False):
    _data = np.random.multivariate_normal(mean, np.eye(n_features) * var, n_samples)
    _data = np.round(np.abs(_data%v_bound)) if do_abs else np.round(_data%v_bound)
    return _data + shift

def gen_feature_vectors(n_columns=2, n_samples=100,
                   features=None,
                   mean=[0], var=[1],
                   min_var=0, max_var=2):
    mean = mean if all([isinstance(mean, list), len(mean)==n_columns]) else [mean[0] for i in range(n_columns)]
    var = var if all([isinstance(var, list), len(var)==n_columns]) else [var[0] for i in range(n_columns)]
    _data = gen_multi_normal_data(n_samples=n_samples, n_features=n_columns,
                                  mean=mean, var=var,
                                  v_bound=max_var, shift=min_var, do_abs=True)
    _data = pd.DataFrame(_data, columns=features)
    return _data

def gen_dummy_data(n_columns=2, n_samples=100,
                   is_positive=False,
                   features=[],
                   mean=[0], var=[1],
                   min_var=0, max_var=2,
                   do_scale=False, scale_func=None):
    if not len(features)==n_columns:
        features = ['f_%d' % (i+1) for i in range(n_columns)]
    _df = gen_feature_vectors(n_columns, n_samples, features, min_var=min_var, max_var=max_var)
    _df = pd.DataFrame((scale_func)(_df), columns=features) if all([do_scale, scale_func]) else _df
    _df['t'] = 1 if is_positive else 0
    return _df

def gen_data_from_csv(path=None, header='infer', sep=','):
    _df = pd.read_csv(path, header=header, sep=sep)
    return _df

def gen_prob_center_list(last=0, cycle_unit=None):
    if not cycle_unit: raise ValueError('Required parameter cycle_unit is undefined.')
    prob_center_list = list()
    for k, v in enumerate(cycle(cycle_unit)):
        if k==last: break
        prob_center_list.append(v)
    return prob_center_list

def gen_dummy_cat_data(n_columns=2, n_samples=100,
                       probs=None, prob_center_start=0,
                       n_options=3, is_positive=True,
                       is_scale=False, scale_func=None):
    df = pd.DataFrame()
    d_probs = deque(probs)
    d_probs.rotate(prob_center_start)
    p_list = [1/n_options] * n_options 
    options = list(range(n_options))
    prob_center_list = np.random.choice(options, n_columns, p_list)
    features = ['f_%d' % (i+1) for i in range(n_columns)]

    for f, p in zip(features, prob_center_list):
        d_probs.rotate(p)
        d_p = list(d_probs)
        data = np.random.multinomial(n_samples, d_p, 1)
        data_list = list()
        for k, v in zip(options, data.flatten()):
            data_list = data_list + [k] * v 
        df[f] = data_list
        d_probs.rotate(-p)
    if all([is_scale, scale_func]):
        df = pd.DataFrame((scale_func)(df), columns=features) if is_scale else df
    df['t'] = 1 if is_positive else 0
    return df
