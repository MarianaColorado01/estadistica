import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy import stats
import os

PLOTS_DIR = os.path.join('static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def _save_fig(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_normal_shaded(x_left, x_right, mean=0, std=1, filename="normal.png"):
    """
    Dibuja la normal estándar y sombrea entre x_left y x_right.
    Si x_left < -5 o x_right > 5, se recortan.
    """
    xs = np.linspace(-5, 5, 1000)
    ys = stats.norm.pdf(xs, mean, std)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, where=(xs >= x_left) & (xs <= x_right))
    ax.set_title("Distribución Normal (Z)")
    ax.set_xlabel("z")
    return _save_fig(fig, filename)

def plot_t_shaded(x_left, x_right, df=10, filename="t.png"):
    xs = np.linspace(-5, 5, 1000)
    ys = stats.t.pdf(xs, df)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, where=(xs >= x_left) & (xs <= x_right))
    ax.set_title(f"Distribución t (df={df})")
    ax.set_xlabel("t")
    return _save_fig(fig, filename)

def plot_chi2_shaded(x_left, x_right, df=4, filename="chi2.png"):
    xs = np.linspace(0, stats.chi2.ppf(0.999, df), 1000)
    ys = stats.chi2.pdf(xs, df)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, where=(xs >= x_left) & (xs <= x_right))
    ax.set_title(f"Distribución χ² (df={df})")
    ax.set_xlabel("χ²")
    return _save_fig(fig, filename)

def plot_f_shaded(x_left, x_right, d1=5, d2=5, filename="f.png"):
    xs = np.linspace(0, stats.f.ppf(0.999, d1, d2), 1000)
    ys = stats.f.pdf(xs, d1, d2)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, where=(xs >= x_left) & (xs <= x_right))
    ax.set_title(f"Distribución F (d1={d1}, d2={d2})")
    ax.set_xlabel("F")
    return _save_fig(fig, filename)


"""Grafico del primer caso: intervalo de confianza para proporción"""
def plot_ci_teorico_y_real(lo, hi, p_hat, se, zcrit, filename="ci_mix.png"):
    
    # Curva normal en la escala real
    xs = np.linspace(p_hat - 4*se, p_hat + 4*se, 2000)
    ys = stats.norm.pdf((xs - p_hat) / se) / se

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(xs, ys)

    # Región del intervalo teórico
    teor_lo = p_hat - zcrit * se
    teor_hi = p_hat + zcrit * se

    # sombreado teórico
    ax.fill_between(xs, 0, ys, where=(xs >= teor_lo) & (xs <= teor_hi),
                    alpha=0.25, color="skyblue", label="Intervalo teórico (Z)")

    # sombreado real
    ax.fill_between(xs, 0, ys, where=(xs >= lo) & (xs <= hi),
                    alpha=0.45, color="violet", label="Intervalo real")

    ax.set_title("Comparación: Intervalo teórico vs Intervalo real")
    ax.set_xlabel("Valor de la proporción")
    ax.legend()
    return _save_fig(fig, filename)

""""Grafico del segundo caso: intervalo de confianza para diferencia de proporciones"""

def plot_ci_diff_prop_teorico_y_real(lo, hi, p1, n1, p2, n2, conf, filename="ci_diff_prop_mix.png"):
    pdiff = p1 - p2
    se = ((p1*(1-p1)/n1) + (p2*(1-p2)/n2)) ** 0.5
    zcrit = stats.norm.ppf(1 - (1 - conf)/2)

    xs = np.linspace(pdiff - 4*se, pdiff + 4*se, 2000)
    ys = stats.norm.pdf((xs - pdiff) / se) / se

    teor_lo = pdiff - zcrit * se
    teor_hi = pdiff + zcrit * se

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(xs, ys)
    ax.fill_between(xs, 0, ys, where=(xs >= teor_lo) & (xs <= teor_hi),
                    alpha=0.25, color="skyblue", label="Intervalo teórico (Z)")
    ax.fill_between(xs, 0, ys, where=(xs >= lo) & (xs <= hi),
                    alpha=0.5, color="violet", label="Intervalo real")

    ax.set_title("IC Diferencia de Proporciones: Teórico vs Real")
    ax.set_xlabel("p1 - p2")
    ax.legend()
    return _save_fig(fig, filename)

""""Grafico del tercer caso: intervalo de confianza para media"""

def plot_ci_media_teorico_y_real(lo, hi, xbar, se, crit, dist="z", filename="ci_media_mix.png"):
    xs = np.linspace(xbar - 4*se, xbar + 4*se, 2000)

    if dist == "z":
        ys = stats.norm.pdf((xs - xbar)/se) / se
    else:
        ys = stats.t.pdf((xs - xbar)/se, df=crit["df"]) / se

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(xs, ys)

    teor_lo = xbar - crit["value"] * se
    teor_hi = xbar + crit["value"] * se

    ax.fill_between(xs, 0, ys, where=(xs >= teor_lo) & (xs <= teor_hi),
                    alpha=0.25, color="skyblue", label=f"Intervalo teórico ({dist.upper()})")
    ax.fill_between(xs, 0, ys, where=(xs >= lo) & (xs <= hi),
                    alpha=0.45, color="violet", label="Intervalo real")

    ax.set_title("IC Media: Teórico vs Real")
    ax.set_xlabel("Valor")
    ax.legend()
    return _save_fig(fig, filename)


def plot_ci_diff_media_teorico_y_real(lo, hi, diff, se, tcrit, df, filename="ci_diff_media_mix.png"):
    xs = np.linspace(diff - 4*se, diff + 4*se, 2000)
    ys = stats.t.pdf((xs - diff)/se, df=df) / se

    teor_lo = diff - tcrit * se
    teor_hi = diff + tcrit * se

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(xs, ys)

    ax.fill_between(xs, 0, ys, where=(xs >= teor_lo) & (xs <= teor_hi),
                    alpha=0.25, color="skyblue", label="Intervalo teórico (t)")
    ax.fill_between(xs, 0, ys, where=(xs >= lo) & (xs <= hi),
                    alpha=0.45, color="violet", label="Intervalo real")

    ax.set_title("IC Diferencia de Medias: Teórico vs Real")
    ax.set_xlabel("x1 - x2")
    ax.legend()
    return _save_fig(fig, filename)