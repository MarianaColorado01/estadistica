import math
from scipy import stats
import numpy as np

def z_critical(conf):
    alpha = 1 - conf
    return stats.norm.ppf(1 - alpha/2)

def t_critical(conf, df):
    alpha = 1 - conf
    return stats.t.ppf(1 - alpha/2, df)

def interval_proporcion(p_hat, n, conf):
    """
    Intervalo para proporción (usando normal)
    p_hat: proporción en [0,1]
    n: tamaño muestra
    conf: nivel (e.g., 0.95)
    """
    z = z_critical(conf)
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    margin = z * se
    return p_hat - margin, p_hat + margin, margin

def interval_diff_proporciones(p1, n1, p2, n2, conf):
    z = z_critical(conf)
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    diff = p1 - p2
    margin = z * se
    return diff - margin, diff + margin, margin

def z_test_proporcion(p_hat, p0, n, alternative='two-sided'):
    se = math.sqrt(p0*(1-p0)/n)
    z = (p_hat - p0) / se
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:
        p_value = stats.norm.cdf(z)
    return z, p_value

def interval_media(xbar, sigma, n, conf):
    """
    Si sigma es None -> t student con df = n-1
    Si sigma es proporcionado -> z
    Returns lo, hi, margin, used ('z' or 't')
    """
    if sigma is None or sigma == '':
        # t
        df = n - 1
        t = t_critical(conf, df)
        s = None
        # here we cannot compute margin without sample sd;
        # assume user will use an input that provides s via different path.
        # To keep this function useful: we expect caller to provide sigma when known,
        # otherwise we can't compute margin without sample sd. So call should have s passed in
        # but to keep interface minimal, raise if sigma none.
        raise ValueError("sigma (poblacional) no proporcionada: use muestra y desviación muestral para t. Use otra función.")
    else:
        z = z_critical(conf)
        se = sigma / math.sqrt(n)
        margin = z * se
        return xbar - margin, xbar + margin, margin, 'z'

def interval_media_from_sample(xbar, s, n, conf):
    df = n - 1
    t = t_critical(conf, df)
    se = s / math.sqrt(n)
    margin = t * se
    return xbar - margin, xbar + margin, margin, 't'

def t_test_media(xbar, n, mu0, alternative='two-sided', s=None):
    """
    If s provided, use sample sd passed as argument? Here we assume caller has sample sd or
    has t-statistic context. For simplicity, if s is not provided we cannot compute t.
    To keep interface used in main.py: we will compute t using provided xbar and s in that flow.
    """
    # this function may be used when s is known in caller
    raise NotImplementedError("Use caller to compute t-stat when sample sd is available.")

def interval_diff_medias_welch(x1, s1, n1, x2, s2, n2, conf):
    # Welch's interval
    df_num = (s1**2 / n1 + s2**2 / n2)**2
    df_den = ( (s1**2 / n1)**2 / (n1 - 1) ) + ( (s2**2 / n2)**2 / (n2 - 1) )
    df = df_num / df_den
    tcrit = stats.t.ppf(1 - (1-conf)/2, df)
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    diff = x1 - x2
    margin = tcrit * se
    return diff - margin, diff + margin, margin

def t_test_diff_medias_welch(x1, s1, n1, x2, s2, n2, alternative='two-sided'):
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t_stat = (x1 - x2) / se
    # degrees of freedom approx
    df_num = (s1**2 / n1 + s2**2 / n2)**2
    df_den = ( (s1**2 / n1)**2 / (n1 - 1) ) + ( (s2**2 / n2)**2 / (n2 - 1) )
    df = df_num / df_den
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df)
    else:
        p_value = stats.t.cdf(t_stat, df)
    return t_stat, p_value