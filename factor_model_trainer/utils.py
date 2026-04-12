import numpy as np
from factor_analyzer import FactorAnalyzer
from scipy.stats import chi2 as chi2_dist


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
        l = factor_model.loadings_
        p = l.shape[0]
        m = l.shape[1]
        psi = np.diag(factor_model.get_uniquenesses())

        sigma = l @ l.T + psi if not is_null else np.eye(p)
        sign_s, logdet_s = np.linalg.slogdet(s)
        sign_m, logdet_m = np.linalg.slogdet(sigma)
        sigma_inv = np.linalg.inv(sigma)

        f_ml = logdet_m + np.trace(s @ sigma_inv) - logdet_s - p
        chi_sq = (n_obs - 1) * f_ml
        df = get_df(p, m)
        p_val = chi2_dist.sf(chi_sq, df)
    except:
        pass

    return chi_sq, p_val
