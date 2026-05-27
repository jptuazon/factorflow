import math
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm, multivariate_normal, kendalltau
from scipy.optimize import minimize_scalar
import warnings
from scipy.stats import binomtest


def get_df(num_man, num_fac):
    return (
        (num_man * (num_man + 1)) / 2 - num_man * num_fac - num_man - (num_fac * (num_fac - 1)) / 2
    )


def get_chi_sq(factor_model, n_obs, is_null=False):
    if not isinstance(factor_model, FactorAnalyzer):
        raise TypeError("factor_model must be a factor_analyzer.factor_analyzer.FactorAnalyzer object")

    chi_sq = None
    p_val = None

    try:
        s = factor_model.corr_
        loadings = factor_model.loadings_
        p = loadings.shape[0]
        m = loadings.shape[1]
        psi = np.diag(factor_model.get_uniquenesses())

        sigma = loadings @ loadings.T + psi if not is_null else np.eye(p)
        sign_s, logdet_s = np.linalg.slogdet(s)
        sign_m, logdet_m = np.linalg.slogdet(sigma)
        sigma_inv = np.linalg.inv(sigma)

        f_ml = logdet_m + np.trace(s @ sigma_inv) - logdet_s - p
        chi_sq = (n_obs - 1) * f_ml
        df = get_df(p, m)
        p_val = chi2_dist.sf(chi_sq, df)
    except Exception:
        pass

    return chi_sq, p_val


def estimate_thresholds(values):
    """
    Adapted from ordinalcorr.

    Original source: https://github.com/nigimitama/ordinalcorr by Masayoshi Mita
    Licensed under MIT.
    """

    values = np.asarray(values)
    levels = np.sort(np.unique(values))
    n = len(values)

    thresholds = []

    for level in levels[:-1]:
        p = (np.sum(values <= level) + 0.5) / (n + 1)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        thresholds.append(norm.ppf(p))

    return np.concatenate((
        np.array([-np.inf]),
        np.asarray(thresholds),
        np.array([np.inf])
    ))


def bivariate_cdf(rho):
    """
    Adapted from ordinalcorr.

    Original source: https://github.com/nigimitama/ordinalcorr by Masayoshi Mita
    Licensed under MIT.
    """

    cov = np.array([[1.0, rho],
                    [rho, 1.0]])

    mvn = multivariate_normal(mean=[0.0, 0.0], cov=cov, allow_singular=True)

    def phi(x):
        return mvn.cdf(x)

    def bvn_cdf(lower, upper):
        return (
            phi(upper)
            - phi([lower[0], upper[1]])
            - phi([upper[0], lower[1]])
            + phi(lower)
        )

    return bvn_cdf


def polychoric(x, y):
    """
    Adapted from ordinalcorr.

    Original source: https://github.com/nigimitama/ordinalcorr by Masayoshi Mita
    Licensed under MIT.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        warnings.warn("x and y must have same length")
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        warnings.warn("Zero variance in input")
        return np.nan

    x_levels = np.sort(np.unique(x))
    y_levels = np.sort(np.unique(y))

    if len(x_levels) < 2 or len(y_levels) < 2:
        warnings.warn("Both variables must have at least 2 levels")
        return np.nan

    tau_x = estimate_thresholds(x)
    tau_y = estimate_thresholds(y)

    x_map = {v: i for i, v in enumerate(x_levels)}
    y_map = {v: j for j, v in enumerate(y_levels)}

    contingency = np.zeros((len(x_levels), len(y_levels)), dtype=int)

    for xi, yi in zip(x, y):
        contingency[x_map[xi], y_map[yi]] += 1

    def neg_log_likelihood(rho):
        if abs(rho) >= 0.99:
            return np.inf

        bvn = bivariate_cdf(rho)
        ll = 0.0

        for i in range(len(x_levels)):
            for j in range(len(y_levels)):

                n_ij = contingency[i, j]
                if n_ij == 0:
                    continue

                lower = [tau_x[i], tau_y[j]]
                upper = [tau_x[i + 1], tau_y[j + 1]]

                p_ij = bvn(lower, upper)
                p_ij = max(p_ij, 1e-12)

                if not np.isfinite(p_ij) or p_ij <= 0:
                    return np.inf

                ll += n_ij * np.log(p_ij)

        return -ll

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(-0.98, 0.98),
        method="bounded"
    )

    return result.x


def get_polychoric_matrix(data_arr):
    if not isinstance(data_arr, np.ndarray):
        raise TypeError("data_arr must be a numpy array")

    var_count = data_arr.shape[1]
    poly_corr_mat = np.eye(var_count)

    for row in range(var_count):
        for col in range(row):
            row_arr = data_arr[:, row]
            col_arr = data_arr[:, col]

            corr = polychoric(row_arr, col_arr)

            poly_corr_mat[row, col] = corr
            poly_corr_mat[col, row] = corr

    return poly_corr_mat


def get_multiset(prior_matrix, loading_sim_matrix, with_labels=False):
    if not isinstance(prior_matrix, np.ndarray):
        raise TypeError("prior_matrix must be a numpy array")
    if not isinstance(loading_sim_matrix, np.ndarray):
        raise TypeError("loading_sim_matrix must be a numpy array")
    if prior_matrix.shape != loading_sim_matrix.shape:
        raise ValueError("prior_matrix and loading_sim_matrix must have the same dimensions")

    num_of_vars = prior_matrix.shape[0]
    x = []
    y = []
    var_1 = []
    var_2 = []
    for i in range(num_of_vars):
        for j in range(i):
            if not pd.isna(prior_matrix[i, j]):
                if with_labels:
                    var_1.append(f"X{i + 1}")
                    var_2.append(f"X{j + 1}")

                x.append(prior_matrix[i, j])
                y.append(loading_sim_matrix[i, j])
    x = np.array(x)
    y = np.array(y)

    if with_labels:
        return x, y, var_1, var_2
    else:
        return x, y


def get_v_index(prior_matrix, loading_sim_matrix):
    if not isinstance(prior_matrix, np.ndarray):
        raise TypeError("prior_matrix must be a numpy array")
    if not isinstance(loading_sim_matrix, np.ndarray):
        raise TypeError("loading_sim_matrix must be a numpy array")
    if prior_matrix.shape != loading_sim_matrix.shape:
        raise ValueError("prior_matrix and loading_sim_matrix must have the same dimensions")

    x, y = get_multiset(prior_matrix, loading_sim_matrix)
    n = len(x)

    theta = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    theta = theta / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    theta = (1 / math.pi) * np.arctan(theta) + 1 / 2

    tau = (1 / 2) * (kendalltau(x, y, variant="b").statistic + 1)

    v_index = math.sqrt(tau * theta)

    return v_index


def sign_test(x, median=0, alternative="greater"):
    diffs = np.array(x) - median
    non_zero_diffs = np.sum(diffs != 0)
    n_plus = np.sum(diffs > 0)

    if non_zero_diffs == 0:
        return None, None

    test = binomtest(
        n_plus,
        non_zero_diffs,
        p=0.5,
        alternative=alternative
    )

    test_stat = n_plus
    p_val = test.pvalue

    return test_stat, p_val
