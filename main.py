from flask import Flask, render_template, request, redirect, url_for
from calculos import (interval_proporcion, interval_diff_proporciones, z_test_proporcion, interval_media, t_test_media, interval_diff_medias_welch, t_test_diff_medias_welch)
from distribuciones import plot_normal_shaded, plot_t_shaded, plot_chi2_shaded, plot_f_shaded
import os
from datetime import datetime
from scipy import stats
from distribuciones import plot_ci_teorico_y_real
from distribuciones import plot_ci_diff_prop_teorico_y_real
from distribuciones import plot_ci_media_teorico_y_real
from distribuciones import plot_ci_diff_media_teorico_y_real

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

PLOTS_DIR = os.path.join('static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

CASES = {
    "1": "Intervalo de confianza - Proporción (1 muestra)",
    "2": "Intervalo de confianza - Diferencia de proporciones (2 muestras)",
    "3": "Prueba de hipótesis - Proporción",
    "4": "Intervalo de confianza - Media (Z o t)",
    "5": "Prueba de hipótesis - Media (Z o t)",
    "6": "Diferencia de medias (Intervalo y prueba)"
}

@app.route('/')
def index():
    return render_template('index.html', cases=CASES)

@app.route('/form/<case_id>')
def form_case(case_id):
    if case_id not in CASES:
        return redirect(url_for('index'))
    return render_template('formulario.html', case_id=case_id, case_title=CASES[case_id])

@app.route('/compute/<case_id>', methods=['POST'])
def compute(case_id):
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    plotfile = None
    result = {}
    interpretation = ""

    if case_id == "1":
        p = float(request.form['p'])
        n = int(request.form['n'])
        conf = float(request.form.get('conf', 95)) / 100.0

        lo, hi, margin = interval_proporcion(p, n, conf)
        result = {"lo": lo, "hi": hi, "margin": margin, "p": p, "n": n, "conf": conf}

        # Gráfica combinada
        se = (p * (1 - p) / n) ** 0.5
        zcrit = stats.norm.ppf(1 - (1 - conf) / 2)

        plotfile = plot_ci_teorico_y_real(
            lo=lo, hi=hi,
            p_hat=p, se=se, zcrit=zcrit,
            filename=f"ci_mix_{ts}.png"
        )
    elif case_id == "2":
        p1 = float(request.form['p1'])
        n1 = int(request.form['n1'])
        p2 = float(request.form['p2'])
        n2 = int(request.form['n2'])
        conf = float(request.form.get('conf', 95)) / 100.0

        lo, hi, margin = interval_diff_proporciones(p1, n1, p2, n2, conf)
        if hi > 1: hi = 1.0

        result = {
            "lo": lo, "hi": hi, "margin": margin,
            "p1": p1, "n1": n1, "p2": p2, "n2": n2, "conf": conf,
            "pdiff": max(min(p1 - p2, 1), -1)
        }

        plotfile = plot_ci_diff_prop_teorico_y_real(
            lo=lo, hi=hi,
            p1=p1, n1=n1, p2=p2, n2=n2,
            conf=conf,
            filename=f"ci_diff_prop_mix_{ts}.png"
        )

    elif case_id == "3":
        p_hat = float(request.form['p_hat'])
        p0 = float(request.form['p0'])
        n = int(request.form['n'])
        alt = request.form['alternative']

        z_stat, p_value = z_test_proporcion(p_hat, p0, n, alternative=alt)

        conf = float(request.form.get('conf', 95)) / 100.0
        alpha = 1 - conf

        # Valor crítico
        if alt == "greater":
            zcrit = stats.norm.ppf(1 - alpha)
        elif alt == "less":
            zcrit = stats.norm.ppf(alpha)
        else:
            zcrit = stats.norm.ppf(1 - alpha/2)

        # Decisión
        if alt == "greater":
            decision = "Rechazar Ho" if z_stat > zcrit else "No rechazar Ho"
        elif alt == "less":
            decision = "Rechazar Ho" if z_stat < zcrit else "No rechazar Ho"
        else:
            decision = "Rechazar Ho" if abs(z_stat) > abs(zcrit) else "No rechazar Ho"

        interpretacion = f"""
        Hipótesis:
        • H0: p = {p0}
        • H1: p {">" if alt=="greater" else "<" if alt=="less" else "≠"} {p0}

        Nivel de significancia α = {alpha:.3f}

        Estadístico Z = {z_stat:.4f}
        Valor crítico Zc = {zcrit:.4f}

        Decisión: {decision}.
        """

        result = {
            "z": z_stat,
            "p_value": p_value,
            "p_hat": p_hat,
            "p0": p0,
            "n": n,
            "alt": alt,
            "conf": conf,
            "zcrit": zcrit,
            "decision": decision,
            "interpretacion": interpretacion
        }

        # Gráfica
        if alt == 'two-sided':
            plotfile = plot_normal_shaded(-abs(z_stat), abs(z_stat), filename=f"normal_test_{ts}.png")
        elif alt == 'greater':
            plotfile = plot_normal_shaded(z_stat, 5, filename=f"normal_test_{ts}.png")
        else:
            plotfile = plot_normal_shaded(-5, z_stat, filename=f"normal_test_{ts}.png")

    elif case_id == "4":
        xbar = float(request.form['xbar'])
        n = int(request.form['n'])
        conf = float(request.form.get('conf', 95)) / 100.0
        sigma = request.form.get('sigma')

        if sigma:  
            sigma = float(sigma)
            lo, hi, margin, used = interval_media(xbar, sigma, n, conf)

            result = {
                "lo": lo, "hi": hi, "margin": margin,
                "xbar": xbar, "n": n, "conf": conf, "used": used
            }

            plotfile = plot_ci_media_teorico_y_real(
                lo=lo, hi=hi,
                xbar=xbar,
                se=sigma / (n**0.5),
                crit={"value": stats.norm.ppf(1 - (1-conf)/2)},
                dist="z",
                filename=f"ci_media_mix_{ts}.png"
            )

        else:

            s = float(request.form['s'])

            se = s / (n**0.5)
            tcrit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
            margin = tcrit * se

            lo = xbar - margin
            hi = xbar + margin

            result = {
                "lo": lo, "hi": hi, "margin": margin,
                "xbar": xbar, "n": n, "conf": conf, "used": "t"
            }

            
            plotfile = plot_ci_media_teorico_y_real(
                lo=lo, hi=hi,
                xbar=xbar,
                se=se,
                crit={"value": tcrit, "df": n-1},
                dist="t",
                filename=f"ci_media_mix_{ts}.png"
            )
            

    elif case_id == "5":
        xbar = float(request.form['xbar'])
        n = int(request.form['n'])
        mu0 = float(request.form['mu0'])
        alt = request.form['alternative']
        sigma = request.form.get('sigma')

        if sigma:
            sigma = float(sigma)
            z_stat = (xbar - mu0) / (sigma / (n**0.5))

            if alt == 'two-sided':
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            elif alt == 'greater':
                p_value = 1 - stats.norm.cdf(z_stat)
            else:
                p_value = stats.norm.cdf(z_stat)

            result = {"stat": z_stat, "p_value": p_value, "method": "z"}

            if alt == 'two-sided':
                plotfile = plot_normal_shaded(-abs(z_stat), abs(z_stat), filename=f"normal_test_{ts}.png")
            elif alt == 'greater':
                plotfile = plot_normal_shaded(z_stat, 5, filename=f"normal_test_{ts}.png")
            else:
                plotfile = plot_normal_shaded(-5, z_stat, filename=f"normal_test_{ts}.png")

        else:
            t_stat, p_value = t_test_media(xbar, n, mu0, alternative=alt)
            result = {"stat": t_stat, "p_value": p_value, "method": "t"}

            if alt == 'two-sided':
                plotfile = plot_t_shaded(-abs(t_stat), abs(t_stat), df=n-1, filename=f"t_test_{ts}.png")
            elif alt == 'greater':
                plotfile = plot_t_shaded(t_stat, 5, df=n-1, filename=f"t_test_{ts}.png")
            else:
                plotfile = plot_t_shaded(-5, t_stat, df=n-1, filename=f"t_test_{ts}.png")
    elif case_id == "6":
        x1 = float(request.form['x1'])
        s1 = float(request.form['s1'])
        n1 = int(request.form['n1'])
        x2 = float(request.form['x2'])
        s2 = float(request.form['s2'])
        n2 = int(request.form['n2'])
        conf = float(request.form.get('conf', 95)) / 100.0

        lo, hi, margin = interval_diff_medias_welch(x1, s1, n1, x2, s2, n2, conf)
        t_stat, p_value = t_test_diff_medias_welch(x1, s1, n1, x2, s2, n2, alternative=request.form['alternative'])

        se = (s1**2 / n1 + s2**2 / n2) ** 0.5

        df_num = (s1**2 / n1 + s2**2 / n2)**2
        df_den = ((s1**2 / n1)**2 / (n1-1)) + ((s2**2 / n2)**2 / (n2-1))
        df = df_num / df_den

        tcrit = stats.t.ppf(1 - (1-conf)/2, df)

        result = {
            "lo": lo, "hi": hi, "margin": margin,
            "x1": x1, "x2": x2, "n1": n1, "n2": n2,
            "conf": conf, "t": t_stat, "p": p_value
        }

        plotfile = plot_ci_diff_media_teorico_y_real(
            lo=lo, hi=hi,
            diff=(x1 - x2),
            se=se,
            tcrit=tcrit,
            df=df,
            filename=f"ci_diff_media_mix_{ts}.png"
        )

    else:
        return redirect(url_for('index'))

    # Convertir numpy floats a float normales
    result = {k: float(v) if hasattr(v, "item") else v for k, v in result.items()}

    return render_template(
        'resultado.html', case_id=case_id, case_title=CASES[case_id], result=result, plotfile=plotfile
    )

if __name__ == '__main__':
    app.run(debug=True)
