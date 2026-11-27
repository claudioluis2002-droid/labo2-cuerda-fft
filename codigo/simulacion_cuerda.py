from pathlib import Path
import csv
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'labo2-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

L = 1.0
T_TENSION = 100.0
MU = 0.01
C = np.sqrt(T_TENSION / MU)
A_AMP = 0.005


def deformacion_triangular(x, a, A=A_AMP, L=L):
    y0 = np.zeros_like(x, dtype=float)
    izq = x <= a
    der = x > a
    y0[izq] = (A / a) * x[izq]
    y0[der] = (A / (L - a)) * (L - x[der])
    y0.put(0, 0.0)
    y0.put(-1, 0.0)
    return y0


def coeficientes_fourier_triangular(n_max, a, A=A_AMP, L=L):
    n = np.arange(1, n_max + 1, dtype=float)
    return (2.0 * A * L**2 / (n**2 * np.pi**2 * a * (L - a))) * np.sin(n * np.pi * a / L)


def solucion_analitica_triangular(x, t, a, n_modos=200, A=A_AMP, L=L, c=C):
    Bn = coeficientes_fourier_triangular(n_modos, a, A=A, L=L)
    n = np.arange(1, n_modos + 1, dtype=float)
    omega_n = n * np.pi * c / L
    senos = np.sin(np.outer(n * np.pi / L, x))
    cosenos = np.cos(np.outer(t, omega_n))
    return (cosenos * Bn) @ senos


def energia_inicial_analitica(a, A=A_AMP, L=L, T=T_TENSION):
    return (T * A**2 * L) / (2.0 * a * (L - a))


def calcular_energia_discreta(y_curr, y_prev, dx, dt, mu=MU, T=T_TENSION):
    v = (y_curr[1:-1] - y_prev[1:-1]) / dt
    e_cin = 0.5 * mu * dx * np.sum(v**2)
    s_curr = (y_curr[1:] - y_curr[:-1]) / dx
    s_prev = (y_prev[1:] - y_prev[:-1]) / dx
    e_pot = 0.5 * T * dx * np.sum(s_curr * s_prev)
    return float(e_cin + e_pot)


def resolver_cuerda_implicita(y0, Nx, dt, n_pasos, L=L, c=C, mu=MU, T=T_TENSION,
                              guardar_todo=False, indices_sensores=None):
    dx = L / Nx
    sigma = c * dt / dx
    alpha = 0.25 * sigma**2
    n_int = Nx - 1

    ab = np.zeros((3, n_int))
    ab[0, 1:] = -alpha
    ab[1, :] = 1.0 + 2.0 * alpha
    ab[2, :-1] = -alpha

    y_prev = y0.copy()
    y_prev.put(0, 0.0)
    y_prev.put(-1, 0.0)

    y_curr = np.zeros_like(y_prev)
    d2_y0 = y_prev[2:] - 2.0 * y_prev[1:-1] + y_prev[:-2]
    rhs0 = y_prev[1:-1] + 0.25 * (sigma**2) * d2_y0
    y_curr[1:-1] = solve_banded((1, 1), ab, rhs0)

    t_vec = np.arange(n_pasos + 1) * dt
    energia = np.zeros(n_pasos)
    energia.put(0, calcular_energia_discreta(y_curr, y_prev, dx, dt, mu=mu, T=T))

    historial = np.zeros((n_pasos + 1, Nx + 1)) if guardar_todo else None
    if guardar_todo:
        historial[0, :] = y_prev
        historial[1, :] = y_curr

    sensores = np.zeros((n_pasos + 1, len(indices_sensores))) if indices_sensores else None
    if indices_sensores:
        sensores[0, :] = y_prev[indices_sensores]
        sensores[1, :] = y_curr[indices_sensores]

    for n in range(1, n_pasos):
        d2_curr = y_curr[2:] - 2.0 * y_curr[1:-1] + y_curr[:-2]
        d2_prev = y_prev[2:] - 2.0 * y_prev[1:-1] + y_prev[:-2]
        rhs = 2.0 * y_curr[1:-1] - y_prev[1:-1] + alpha * (2.0 * d2_curr + d2_prev)
        y_next = np.zeros_like(y_curr)
        y_next[1:-1] = solve_banded((1, 1), ab, rhs)
        energia[n] = calcular_energia_discreta(y_next, y_curr, dx, dt, mu=mu, T=T)
        if guardar_todo:
            historial[n + 1, :] = y_next
        if indices_sensores:
            sensores[n + 1, :] = y_next[indices_sensores]
        y_prev = y_curr
        y_curr = y_next

    return {'t': t_vec, 'y_final': y_curr, 'energia': energia,
            'historial': historial, 'sensores': sensores, 'sigma': sigma}


def verificar_modo_unico(salida, figuras):
    t_total = 0.20
    omega1 = np.pi * C / L
    Nx_base = 200
    dt_base = 1.0e-4
    n_pasos_base = int(round(t_total / dt_base))
    x_base = np.linspace(0, L, Nx_base + 1)
    y0_base = A_AMP * np.sin(np.pi * x_base / L)

    res_base = resolver_cuerda_implicita(y0_base, Nx_base, dt_base, n_pasos_base, guardar_todo=True)
    t_vec = res_base['t']
    y_num_centro = res_base['historial'][:, Nx_base // 2]
    y_exa_centro = A_AMP * np.cos(omega1 * t_vec)

    e_vec = res_base['energia']
    e0_val = float(e_vec.item(0))
    deriva_energia_rel = float(np.max(np.abs(e_vec - e0_val)) / e0_val)

    idx_ult = t_vec >= (t_total - 0.04)
    t_u = t_vec[idx_ult]
    y_u = y_num_centro[idx_ult]
    M_ls = np.column_stack((np.cos(omega1 * t_u), np.sin(omega1 * t_u)))
    coef, _, _, _ = np.linalg.lstsq(M_ls, y_u, rcond=None)
    c0, c1 = float(coef.item(0)), float(coef.item(1))
    amp_num = float(np.hypot(c0, c1))
    fase_num = float(np.arctan2(-c1, c0))
    err_amp_rel = abs(amp_num - A_AMP) / A_AMP
    err_fase_rad = abs(fase_num)

    refinamiento = []
    for Nx in (50, 100, 200, 400, 800):
        dx = L / Nx
        dt = 0.8 * dx / C
        n_pasos = int(round(t_total / dt))
        x = np.linspace(0, L, Nx + 1)
        y0 = A_AMP * np.sin(np.pi * x / L)
        sim = resolver_cuerda_implicita(y0, Nx, dt, n_pasos)
        y_exacta_final = A_AMP * np.sin(np.pi * x / L) * np.cos(omega1 * sim['t'][-1])
        err_l2 = float(np.sqrt(dx * np.sum((sim['y_final'] - y_exacta_final)**2)))
        err_max = float(np.max(np.abs(sim['y_final'] - y_exacta_final)))
        refinamiento.append((Nx, dx, dt, sim['sigma'], err_l2, err_max))

    guardar_csv(salida / 'verificacion_refinamiento.csv',
                ('Nx', 'dx_m', 'dt_s', 'sigma', 'error_L2_m', 'error_max_m'), refinamiento)
    guardar_csv(salida / 'verificacion_energia.csv',
                ('t_s', 'energia_discreta_J', 'variacion_relativa'),
                zip(t_vec[:-1], e_vec, (e_vec - e0_val) / e0_val))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax0, ax1 = axes.flat
    ax0.plot(t_vec * 1000, y_exa_centro * 1000, 'k-', lw=1.4, label='Analítica exacta')
    ax0.plot(t_vec * 1000, y_num_centro * 1000, '--', color='#174c70', lw=1.3,
             label=f'Implícito ($\\sigma={res_base["sigma"]:.1f}$)')
    ax0.set(xlabel='Tiempo (ms)', ylabel='Desplazamiento en $x=L/2$ (mm)',
            title='Evolución del modo fundamental ($n=1$)', xlim=(0, 100))
    ax0.legend(fontsize=8.5, loc='upper right')
    ax0.grid(alpha=0.25)

    dxs = np.array(refinamiento)[:, 1]
    errs = np.array(refinamiento)[:, 4]
    ax1.loglog(dxs * 1000, errs * 1000, 'o-', color='#bb5533', lw=1.5, label='Error $L_2$ numérico')
    ref_pend = errs.item(0) * (dxs / dxs.item(0))**4
    ax1.loglog(dxs * 1000, ref_pend * 1000, 'k:', lw=1.2, label='Referencia $\\mathcal{O}(\\Delta x^4)$')
    ax1.set(xlabel='Paso espacial $\\Delta x$ (mm)', ylabel='Error $L_2$ en $t=0.20$ s (mm)',
            title='Superconvergencia a $\\sigma = 0.8$ fijo')
    ax1.legend(fontsize=8.5, loc='upper left')
    ax1.grid(True, which='both', alpha=0.25)


    fig.tight_layout()
    fig.savefig(figuras / 'fig1_verificacion_modo1.pdf')
    fig.savefig(figuras / 'fig1_verificacion_modo1.png')
    plt.close(fig)

    orden_est = float(np.polyfit(np.log(dxs), np.log(errs), 1).item(0))
    return {'sigma_base': res_base['sigma'], 'energia_inicial_J': e0_val,
            'deriva_energia_relativa_max': deriva_energia_rel,
            'error_amplitud_relativo': err_amp_rel, 'error_fase_rad': err_fase_rad,
            'orden_convergencia_estimado': orden_est}


def calcular_espectro_fft(senal, dt):
    N = len(senal)
    ventana = np.hanning(N)
    factor_coherente = np.mean(ventana)
    senal_win = (senal - np.mean(senal)) * ventana
    fft_vals = np.fft.rfft(senal_win)
    freqs = np.fft.rfftfreq(N, d=dt)
    amplitudes = (2.0 / N) * np.abs(fft_vals) / factor_coherente
    amplitudes.put(0, amplitudes.item(0) * 0.5)
    return freqs, amplitudes


def ejecutar_estudio_pulsacion(salida, figuras):
    Nx = 660
    dt = 1.0e-4
    t_total = 1.0
    n_pasos = int(round(t_total / dt))
    x = np.linspace(0, L, Nx + 1)

    casos_a = (
        {'etiqueta': 'a_1_2', 'nombre': '$a/L = 1/2$', 'frac': 0.5},
        {'etiqueta': 'a_1_3', 'nombre': '$a/L = 1/3$', 'frac': 1.0 / 3.0},
        {'etiqueta': 'a_2_5', 'nombre': '$a/L = 2/5$', 'frac': 0.4},
    )
    sensores_info = (
        {'id': 's_1_2', 'nombre': 'Sensor $x/L = 1/2$', 'frac': 0.5, 'idx': int(round(0.5 * Nx))},
        {'id': 's_1_3', 'nombre': 'Sensor $x/L = 1/3$', 'frac': 1.0 / 3.0, 'idx': int(round((1.0 / 3.0) * Nx))},
        {'id': 's_3_11', 'nombre': 'Sensor $x/L = 3/11$', 'frac': 3.0 / 11.0, 'idx': int(round((3.0 / 11.0) * Nx))},
    )
    indices_sensores = list(s['idx'] for s in sensores_info)
    resumen_casos = {}
    tabla_coeficientes = []
    n_modos_tabla = 10
    bn_dict = {c['etiqueta']: coeficientes_fourier_triangular(n_modos_tabla, c['frac'] * L) for c in casos_a}

    for n_idx in range(n_modos_tabla):
        fn = (n_idx + 1) * C / (2.0 * L)
        tabla_coeficientes.append((n_idx + 1, fn, float(bn_dict['a_1_2'].item(n_idx)) * 1000,
                                   float(bn_dict['a_1_3'].item(n_idx)) * 1000,
                                   float(bn_dict['a_2_5'].item(n_idx)) * 1000))

    guardar_csv(salida / 'coeficientes_modales_Bn.csv',
                ('modo_n', 'frecuencia_Hz', 'Bn_a12_mm', 'Bn_a13_mm', 'Bn_a25_mm'), tabla_coeficientes)

    resultados_sim = {}
    for caso in casos_a:
        a_val = caso['frac'] * L
        y0 = deformacion_triangular(x, a_val)
        sim = resolver_cuerda_implicita(y0, Nx, dt, n_pasos, indices_sensores=indices_sensores)
        x_sens = x[indices_sensores]
        y_ref_sens = solucion_analitica_triangular(x_sens, sim['t'], a_val, n_modos=200)
        rms_sensores = tuple(float(np.sqrt(np.mean((sim['sensores'][:, k] - y_ref_sens[:, k])**2))) for k in range(3))
        e0_analitica = energia_inicial_analitica(a_val)
        e0_discreta = float(sim['energia'].item(0))
        deriva_e = float(np.max(np.abs(sim['energia'] - e0_discreta)) / e0_discreta)

        guardar_csv(salida / f"senales_{caso['etiqueta']}.csv",
                    ('t_s', 'y_x12_num_m', 'y_x13_num_m', 'y_x311_num_m', 'y_x12_ref_m', 'y_x13_ref_m', 'y_x311_ref_m'),
                    np.column_stack((sim['t'], sim['sensores'], y_ref_sens)))

        espectros = {}
        fft_cols = []
        for k, s_info in enumerate(sensores_info):
            freqs, amps = calcular_espectro_fft(sim['sensores'][:, k], dt)
            espectros[s_info['id']] = (freqs, amps)
            if k == 0:
                fft_cols.append(freqs)
            fft_cols.append(amps)

        guardar_csv(salida / f"fft_{caso['etiqueta']}.csv",
                    ('frecuencia_Hz', 'amp_x12_m', 'amp_x13_m', 'amp_x311_m'), np.column_stack(fft_cols))

        resultados_sim[caso['etiqueta']] = {'t': sim['t'], 'sensores': sim['sensores'],
                                            'y_ref_sens': y_ref_sens, 'espectros': espectros}
        resumen_casos[caso['etiqueta']] = {
            'a_sobre_L': caso['frac'], 'energia_inicial_analitica_J': e0_analitica,
            'energia_inicial_discreta_J': e0_discreta, 'deriva_energia_relativa': deriva_e,
            'rms_vs_analitica_mm': tuple(r * 1000 for r in rms_sensores),
        }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax0, ax1 = axes.flat
    modos_n = np.arange(1, n_modos_tabla + 1)
    ancho = 0.26
    ax0.bar(modos_n - ancho, np.abs(bn_dict['a_1_2']) * 1000, width=ancho, color='#174c70', label='$a/L = 1/2$')
    ax0.bar(modos_n, np.abs(bn_dict['a_1_3']) * 1000, width=ancho, color='#bb5533', label='$a/L = 1/3$')
    ax0.bar(modos_n + ancho, np.abs(bn_dict['a_2_5']) * 1000, width=ancho, color='#2e7d32', label='$a/L = 2/5$')
    ax0.set(xlabel='Modo normal $n$', ylabel='Amplitud modal $|B_n|$ (mm)', title='Excitación modal según pulsación', xticks=modos_n)
    ax0.legend(fontsize=8.5)
    ax0.grid(axis='y', alpha=0.25)

    a_grid = np.linspace(0.08, 0.92, 200) * L
    ax1.plot(a_grid / L, energia_inicial_analitica(a_grid) * 1000, 'k-', lw=1.5, label='$E_0(a)$ analítica')
    for caso, col in zip(casos_a, ('#174c70', '#bb5533', '#2e7d32')):
        e_pt = resumen_casos[caso['etiqueta']]['energia_inicial_discreta_J']
        ax1.plot(caso['frac'], e_pt * 1000, 'o', color=col, ms=7, label=f"{caso['nombre']}: {e_pt*1000:.2f} mJ")
    ax1.set(xlabel='Posición relativa $a/L$', ylabel='Energía inicial $E_0$ (mJ)', title='Energía inicial según punto de pulsación')
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figuras / 'fig2_excitacion_y_energia.pdf')
    fig.savefig(figuras / 'fig2_excitacion_y_energia.png')
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), sharex=True, sharey='row')
    colores_sensor = ('#174c70', '#bb5533', '#2e7d32')
    f_modos = np.arange(1, 11) * C / (2.0 * L)
    for row, caso in enumerate(casos_a):
        et = caso['etiqueta']
        for col, s_info in enumerate(sensores_info):
            ax_ij = axes[row, col]
            freqs, amps = resultados_sim[et]['espectros'][s_info['id']]
            mask_f = freqs <= 520.0
            ax_ij.plot(freqs[mask_f], amps[mask_f] * 1000, color=colores_sensor[col], lw=1.2)
            for fm in f_modos:
                ax_ij.axvline(fm, color='0.75', ls=':', lw=0.7, zorder=0)
            ax_ij.grid(alpha=0.2)
            if row == 0:
                ax_ij.set_title(s_info['nombre'], fontsize=10, fontweight='bold')
            if col == 0:
                ax_ij.set_ylabel(f"{caso['nombre']}\nAmplitud FFT (mm)", fontsize=9)
            if row == 2:
                ax_ij.set_xlabel('Frecuencia $f$ (Hz)', fontsize=9)
            ax_ij.set_xlim(0, 520)
    fig.tight_layout()
    fig.savefig(figuras / 'fig3_espectros_observabilidad.pdf')
    fig.savefig(figuras / 'fig3_espectros_observabilidad.png')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.0), sharex=True, gridspec_kw={'height_ratios': (2.2, 1)})
    ax0, ax1 = axes.flat
    t_ms = resultados_sim['a_2_5']['t'] * 1000
    y_num = resultados_sim['a_2_5']['sensores'][:, 2] * 1000
    y_ref = resultados_sim['a_2_5']['y_ref_sens'][:, 2] * 1000
    mask_t = t_ms <= 80.0
    ax0.plot(t_ms[mask_t], y_ref[mask_t], 'k-', lw=1.5, label='Referencia analítica (200 modos)')
    ax0.plot(t_ms[mask_t], y_num[mask_t], '--', color='#2e7d32', lw=1.3, label='Esquema implícito ($N_x=660$)')
    ax0.set(ylabel='Desplazamiento (mm)', title='Señal en sensor no nodal $x/L = 3/11$ para $a/L = 2/5$',
            ylim=(-2.8, 4.3))
    ax0.legend(fontsize=8.5, loc='upper right')
    ax0.grid(alpha=0.25)
    ax1.plot(t_ms[mask_t], (y_num - y_ref)[mask_t], color='#bb5533', lw=1.0)
    ax1.axhline(0, color='0.5', lw=0.7)
    ax1.set(xlabel='Tiempo (ms)', ylabel='Numérico - Ref (mm)')

    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figuras / 'fig4_comparacion_temporal.pdf')
    fig.savefig(figuras / 'fig4_comparacion_temporal.png')
    plt.close(fig)

    return resumen_casos


def guardar_csv(ruta, columnas, filas):
    with ruta.open('w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        escritor.writerow(columnas)
        escritor.writerows(filas)


def ejecutar_todo():
    raiz = Path(__file__).resolve().parent.parent
    salida = raiz / 'resultados'
    figuras = raiz / 'figuras'
    salida.mkdir(exist_ok=True)
    figuras.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
    print('1. Ejecutando verificacion...')
    res_verif = verificar_modo_unico(salida, figuras)
    print('2. Ejecutando simulaciones y FFT...')
    res_casos = ejecutar_estudio_pulsacion(salida, figuras)
    resumen_total = {'parametros_fisicos': {'L_m': L, 'T_N': T_TENSION, 'mu_kg_m': MU, 'c_m_s': C, 'A_m': A_AMP},
                     'verificacion_modo_unico': res_verif, 'casos_pulsacion': res_casos}
    (salida / 'resumen.json').write_text(json.dumps(resumen_total, indent=2) + '\n', encoding='utf-8')
    print('--- Simulacion completada con exito ---')
    print(json.dumps(resumen_total, indent=2))


if __name__ == '__main__':
    ejecutar_todo()
