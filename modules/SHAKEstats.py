"""
SHAKEstat.py
============

Statistical analysis and visualization module for SHAKEtime-based ground motion datasets.

This module provides a suite of tools for analyzing seismic instrument records and
ShakeMap-derived ground motion data using classical and modern statistical techniques.
It is designed for integration with the `SHAKEtime` class, but functions can also be run independently.

Included functionality:
------------------------
1. **Attenuation Plotting** (`plotdata_attenuation`):
   - Log-log or linear plots of ground motion parameters vs distance (e.g., PGA, MMI)
   - Supports nonlinear power-law fitting with optional bootstrap confidence intervals
   - Optional log-scale transformation and custom tick labeling
   - Color-coded marker fill based on EMS-98 Vs30 classes
   - Edge color control using number of responses (`nresp`)
   - Custom marker style, size, linewidth, and legend flexibility

2. **Residual Analysis Plotting** (`plotdata_residuals`):
   - Residuals computed from observed vs fitted values using nonlinear models
   - Color-coded marker fill (Vs30 class) and edge (nresp)
   - Custom tick labels, marker appearance, and soil/nresp-based legend breakdown

3. **Empirical CDF Plotting** (`plotdata_empirical_cdf`):
   - ECDFs with optional overlays of theoretical CDFs (normal, lognormal, gamma, etc.)
   - Automatic or user-defined threshold annotation
   - Kolmogorov-Smirnov (KS) test to find best-fitting distribution
   - Conditional marker coloring by `nresp` threshold
   - Fully customizable marker style, axis label, legend position, and more

4. **Histogram and PDF Fitting** (`plotdata_histogram`):
   - Histogram and fitted probability density functions (PDFs) for intensity measures
   - Fits up to 9 distributions: normal, lognormal, gamma, Weibull, beta, Burr, Pearson III, GEV, etc.
   - Conditional histogram and PDF curve splitting by `nresp` threshold (≥3 vs <3)
   - Clear legend with unified color+linestyle representation for readability
   - Log-transform support, unit-aware labels, and distribution scoring (AIC, BIC, KS)

5. **Quantile-Quantile and Probability-Probability Plotting** (`plotdata_qq_pp`):
   - QQ and PP plots with best-fit or user-specified distribution
   - Log-transform for skewed distributions (if enabled)
   - Custom styling: marker type, marker color, marker size, line width, and color

Helper functions:
-----------------
- `assign_soil_class_and_color`: Maps Vs30 values to EMS-98 classes and assigns representative colors
- `attenuation_nonlinear_model`: Power-law model with distance saturation for attenuation fitting
- `bootstrap_confidence_band_refactored`: Resampling-based confidence intervals (residual, loess, parametric, bayesian)
- Standardized `Line2D` legend elements for consistent marker styling across all plots

Typical usage (via SHAKEtime):
------------------------------
    shake.plot_attenuation(df, fit='nonlinear', vs30_col='vs30', nresp='n_response')
    shake.plot_empirical_cdf(df, column='pga', compare_dists=['lognorm'], nresp='nresp')
    shake.plot_histogram(df, column='MMI', unit='MMI', nresp='No. of responses')
    shake.plot_qq_pp(df, column='pga', dist_name='lognorm', marker='^', marker_color='green')

Authors:
--------
This module was developed and maintained as part of the SHAKEtime seismic toolkit.

"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
from matplotlib import patches
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.api import OLS, add_constant
from statsmodels.graphics.gofplots import qqplot, ProbPlot
import statsmodels.api as sm
import matplotlib.patches as mpatches
from scipy.stats import t
from matplotlib.lines import Line2D
import itertools

# =============================================================================
# Ground Motion Attenuation Plotting Function
# =============================================================================



def attenuation_nonlinear_model(r, a, b, c):
    return a + b * np.log10(np.maximum(r + c, 1e-6))


def bootstrap_confidence_band_refactored(x_vals, y_vals, fit_func, log_y=True, n_bootstrap=1000, alpha=0.05, method='residual'):
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)
    x_sorted = np.sort(x_vals)

    if method == 'residual':
        try:
            popt, _ = curve_fit(fit_func, x_vals, y_vals, p0=[1, -1, 1], maxfev=10000)
        except Exception as e:
            raise RuntimeError(f"Initial fit failed: {e}")
        y_fit = fit_func(x_vals, *popt)
        residuals = y_vals - y_fit

        y_bootstrap = np.zeros((n_bootstrap, len(x_sorted)))
        for i in range(n_bootstrap):
            resampled_resid = np.random.choice(residuals, size=len(y_vals), replace=True)
            y_boot = y_fit + resampled_resid
            try:
                popt_i, _ = curve_fit(fit_func, x_vals, y_boot, p0=popt, maxfev=10000)
                y_pred = fit_func(x_sorted, *popt_i)
                y_bootstrap[i] = y_pred
            except:
                y_bootstrap[i] = np.nan

        lower = np.nanpercentile(y_bootstrap, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(y_bootstrap, 100 * (1 - alpha / 2), axis=0)
        return x_sorted, 10 ** lower if log_y else lower, 10 ** upper if log_y else upper

    elif method == 'binwise':
        bins = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        y_bootstrap_bins = []

        for i in range(n_bootstrap):
            idxs = np.random.choice(np.arange(len(x_vals)), size=len(x_vals), replace=True)
            x_sample = x_vals[idxs]
            y_sample = y_vals[idxs]
            try:
                popt, _ = curve_fit(fit_func, x_sample, y_sample, p0=[1, -1, 1], maxfev=10000)
                y_pred_all = fit_func(x_vals, *popt)
                bin_values = []
                for j in range(len(bins) - 1):
                    bin_mask = (x_vals >= bins[j]) & (x_vals < bins[j + 1])
                    bin_y = y_sample[bin_mask]
                    bin_values.append(np.nanmean(bin_y) if len(bin_y) >= 3 else np.nan)
                y_bootstrap_bins.append(bin_values)
            except:
                y_bootstrap_bins.append([np.nan] * (len(bins) - 1))

        y_bootstrap_bins = np.array(y_bootstrap_bins)
        lower = np.nanpercentile(y_bootstrap_bins, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(y_bootstrap_bins, 100 * (1 - alpha / 2), axis=0)
        return bin_centers, 10 ** lower if log_y else lower, 10 ** upper if log_y else upper

    elif method == 'parametric':
        try:
            popt, pcov = curve_fit(fit_func, x_vals, y_vals, p0=[1, -1, 1], maxfev=10000)
            dof = len(x_vals) - len(popt)
            t_val = t.ppf(1 - alpha / 2, dof)
            x_sorted = np.sort(x_vals)
            y_fit = fit_func(x_sorted, *popt)
            eps = 1e-8
            J = np.zeros((len(x_sorted), len(popt)))
            for i in range(len(popt)):
                popt_eps = popt.copy()
                popt_eps[i] += eps
                J[:, i] = (fit_func(x_sorted, *popt_eps) - y_fit) / eps
            var_pred = np.sum(J @ pcov * J, axis=1)
            delta = np.clip(t_val * np.sqrt(var_pred), a_min=0, a_max=5)
            lower = y_fit - delta
            upper = y_fit + delta
            return x_sorted, 10 ** lower if log_y else lower, 10 ** upper if log_y else upper
        except:
            return x_vals, np.full_like(x_vals, np.nan), np.full_like(x_vals, np.nan)

    elif method == 'loess':
        x_ref = np.linspace(np.min(x_vals), np.max(x_vals), 200)
        loess_band = []
        for i in range(n_bootstrap):
            idx = np.random.choice(len(x_vals), len(x_vals), replace=True)
            smoothed = lowess(y_vals[idx], x_vals[idx], frac=0.3, xvals=x_ref, return_sorted=False)
            loess_band.append(smoothed)
        loess_band = np.array(loess_band)
        lower = np.nanpercentile(loess_band, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(loess_band, 100 * (1 - alpha / 2), axis=0)
        return x_ref, 10 ** lower if log_y else lower, 10 ** upper if log_y else upper

    elif method == 'bayesian':
        try:
            popt, _ = curve_fit(fit_func, x_vals, y_vals, p0=[1, -1, 1], maxfev=10000)
            y_fit = fit_func(x_vals, *popt)
            residuals = y_vals - y_fit
            sigma_hat = np.std(residuals)
            x_sorted = np.sort(x_vals)
            y_samples = np.zeros((n_bootstrap, len(x_sorted)))
            for i in range(n_bootstrap):
                noise = np.random.normal(0, sigma_hat, size=len(x_sorted))
                y_pred = fit_func(x_sorted, *popt) + noise
                y_samples[i] = y_pred
            lower = np.nanpercentile(y_samples, 100 * alpha / 2, axis=0)
            upper = np.nanpercentile(y_samples, 100 * (1 - alpha / 2), axis=0)
            return x_sorted, 10 ** lower if log_y else lower, 10 ** upper if log_y else upper
        except:
            return x_vals, np.full_like(x_vals, np.nan), np.full_like(x_vals, np.nan)

    else:
        raise ValueError("method must be 'residual', 'binwise', 'parametric', 'loess', or 'bayesian'")



def assign_soil_class_and_color(vs30_vals):
    boundaries = [0, 90, 180, 360, 800, 1500]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    soil_class = []
    soil_color = []
    for vs in vs30_vals:
        if np.isnan(vs):
            soil_class.append('NA')
            soil_color.append('gray')
        elif vs < 90:
            soil_class.append('E')
            soil_color.append('red')
        elif vs < 180:
            soil_class.append('D')
            soil_color.append('orange')
        elif vs < 360:
            soil_class.append('C')
            soil_color.append('yellow')
        elif vs < 800:
            soil_class.append('B')
            soil_color.append('lightgreen')
        else:
            soil_class.append('A')
            soil_color.append('green')
    return np.array(soil_class), np.array(soil_color)





def plotdata_attenuation(
    df, x='distance', y='pga', vs30_col=None,
    log_x=True, log_y=True, unit_y='g', unit_x='km',
    fit='nonlinear', show_confidence=True, output_path=None,
    event_id='event', name='attenuation', save_formats=['png'],
    dpi=300, figsize=(14, 10), show_title=True,
    alpha=0.6, s=50, color='steelblue', marker='o',
    n_bootstrap=1000, ci_method='residual', return_values=False,
    x_ticks=None, y_ticks=None, legend_marker_size=20, legend_loc='best',
    x_label=None, y_label=None, nresp=None, markeredge_width=3
):
    """
    Plot ground motion attenuation with optional regression fit, soil classification, and edge color variation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing distance and ground motion columns.
    x, y : str
        Column names for distance and ground motion (e.g., 'distance', 'pga').
    vs30_col : str, optional
        Column containing Vs30 values for soil classification (EMS-98).
    log_x, log_y : bool
        Use logarithmic scaling for x and/or y axes.
    unit_x, unit_y : str
        Unit labels for x and y axes.
    fit : str
        Type of fit: 'nonlinear', 'loess', etc.
    show_confidence : bool
        Whether to display bootstrap confidence interval.
    output_path : str, optional
        If provided, saves the figure to this path.
    event_id : str
        Earthquake event ID.
    name : str
        Figure file name.
    x_ticks, y_ticks : list, optional
        Custom axis tick locations.
    x_label, y_label : str, optional
        Override the default x and y axis labels.
    nresp : str, optional
        Column name for number of responses to vary edge color (e.g., black if <3, red if ≥3).

    Returns
    -------
    dict
        If return_values is True, returns fitted parameters and residuals.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Prepare data
    cols = [x, y]
    if vs30_col:
        cols.append(vs30_col)
    if nresp:
        cols.append(nresp)
    data = df[cols].dropna()
    x_vals = data[x].values
    y_vals = data[y].values

    # Fill color based on Vs30
    if vs30_col:
        vs30_vals = data[vs30_col].values
        soil_classes, soil_colors = assign_soil_class_and_color(vs30_vals)
        data['soil_class'] = soil_classes
        data['soil_color'] = soil_colors
        fill_colors = soil_colors
    else:
        fill_colors = color

    # Edge color based on nresp
    if nresp:
        edge_colors = ['red' if val >= 3 else 'black' for val in data[nresp]]
    else:
        edge_colors = 'k'

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x_vals, y_vals,
        c=fill_colors,
        edgecolors=edge_colors,
        linewidth=markeredge_width,
        s=s,
        alpha=alpha,
        marker=marker
    )

    # Axis labels
    xlabel = x_label if x_label else f"{x} [{unit_x}]" if unit_x else x
    ylabel = y_label if y_label else f"{y} [{unit_y}]" if unit_y else y
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Log scaling
    if log_x:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')

    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')

    # Fit and confidence interval
    results = {}
    if fit == 'nonlinear':
        try:
            y_plot = np.log10(y_vals) if log_y else y_vals
            popt, _ = curve_fit(attenuation_nonlinear_model, x_vals, y_plot, p0=[1, -1, 1], maxfev=10000)
            x_sorted = np.sort(x_vals)
            y_fit = attenuation_nonlinear_model(x_sorted, *popt)
            ax.plot(x_sorted, 10 ** y_fit if log_y else y_fit,
                    linestyle='--', color='orange', label='Nonlinear Fit', linewidth=4)
            if show_confidence:
                x_band, lower, upper = bootstrap_confidence_band_refactored(
                    x_vals, y_plot, attenuation_nonlinear_model,
                    log_y=log_y, n_bootstrap=n_bootstrap, method=ci_method
                )
                ax.fill_between(x_band, lower, upper, color='orange', alpha=0.25, label=f'95% CI ({ci_method})')
            if return_values:
                results['params'] = popt
                results['residuals'] = y_plot - attenuation_nonlinear_model(x_vals, *popt)
        except Exception as e:
            print(f"⚠️ Nonlinear fitting failed: {e}")

    # Title
    if show_title:
        ax.set_title(f"{y} vs {x} (Attenuation)")

    # Custom ticks
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)

    # Legend construction
    legend_elements = []
    if vs30_col:
        soil_classes = ['A', 'B', 'C', 'D', 'E']
        soil_colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        legend_elements.extend([
            Line2D([0], [0], marker=marker, linestyle='None',
                   label=f'Soil Class {cls}', color='none',
                   markerfacecolor=col, markeredgecolor='k',
                   markersize=legend_marker_size, markeredgewidth=markeredge_width)
            for cls, col in zip(soil_classes, soil_colors)
        ])
    if nresp:
        legend_elements.extend([
            Line2D([0], [0], marker=marker, linestyle='None',
                   label='nresp ≥ 3', color='none',
                   markerfacecolor='gray', markeredgecolor='red',
                   markersize=legend_marker_size, markeredgewidth=markeredge_width),
            Line2D([0], [0], marker=marker, linestyle='None',
                   label='nresp < 3', color='none',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=legend_marker_size, markeredgewidth=markeredge_width),
        ])

    if legend_elements:
        current_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=current_handles + legend_elements, loc=legend_loc)
    else:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)

    # Finalize
    #ax.grid(True)
    plt.tight_layout()

    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(out_dir / f"{event_id}_{name}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.show()

    if return_values:
        return results







# =============================================================================
# Empirical CDF Plot Function
# =============================================================================
def plotdata_empirical_cdf(
    df,
    column='pga',
    nresp=None,
    log_transform=False,
    compare_dists=None,
    mark_thresholds=None,
    xscale='linear',
    unit=None,
    output_path=None,
    event_id='event',
    name='ecdf',
    save_formats=['png', 'pdf'],
    dpi=300,
    show_title=True,
    figsize=(24, 12),
    marker_size=6,
    marker_color='black',
    compute_best_fit=True,
    cdf_marker='o',
    linewidth=5,
    legend_loc='best',
    x_label=None,
    y_label=None,
    ):

    """
    Plot an empirical cumulative distribution function (ECDF) with optional
    overlays of theoretical cumulative distribution functions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the column to analyze.
    column : str, default='pga'
        Column name containing the ground motion values.
    log_transform : bool, default=False
        If True, applies log-transform to the column values before plotting.
    compare_dists : list of str, optional
        List of distributions to compare against ECDF (e.g., ['lognorm', 'gamma']).
    mark_thresholds : float or list of float, optional
        X-value(s) to annotate on the ECDF plot.
    xscale : {'linear', 'log'}, default='linear'
        Scale of the x-axis.
    unit : str, optional
        Unit of measurement to annotate axis label.
    output_path : str or Path, optional
        Folder where plots will be saved.
    event_id : str, default='event'
        Event name used in output file paths.
    name : str, default='ecdf'
        File name prefix.
    save_formats : list of str, default=['png', 'pdf']
        File types to save (e.g., ['png', 'svg']).
    dpi : int, default=300
        Resolution of saved plots.
    show_title : bool, default=True
        Whether to show a title on the plot.
    figsize : tuple, default=(24, 12)
        Size of the plot.
    marker_size : int, default=6
        Size of ECDF data point markers.
    marker_color : str, default='black'
        Global color override. If set, disables nresp-based color logic.
    compute_best_fit : bool, default=True
        If True, performs KS test to report best-fitting distribution.
    cdf_marker : str, default='o'
        Marker type for ECDF data points.
    linewidth : float, default=5
        Line width of fitted CDF overlays.
    legend_loc : str, default='best'
        Position of the legend box.
    x_label : str, optional
        Custom label to override automatic x-axis label.
    y_label : str, optional
        Custom label to override automatic y-axis label.
    nresp : str, optional
        Column name in `df` containing number of responses. Points with
        nresp >= 3 are plotted in black; otherwise gray. Disabled if
        marker_color is manually set.

    Returns
    -------
    dict
        Dictionary with keys: 'best_fit', 'ks_stat', 'all_fits'.
    """



    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy import stats
    from matplotlib.ticker import ScalarFormatter
    import itertools
    from matplotlib.lines import Line2D

    all_dists = {
        'normal': (stats.norm, 'norm'),
        'lognorm': (stats.lognorm, 'lognorm'),
        'gamma': (stats.gamma, 'gamma'),
        'weibull': (stats.weibull_min, 'weibull_min'),
        'expon': (stats.expon, 'expon'),
        'beta': (stats.beta, 'beta'),
        'genextreme': (stats.genextreme, 'genextreme'),
        'pearson3': (stats.pearson3, 'pearson3'),
        'burr': (stats.burr12, 'burr12')
    }

    if compare_dists is None:
        compare_dists = list(all_dists.keys())

    if isinstance(mark_thresholds, (int, float)):
        mark_thresholds = [mark_thresholds]
    elif mark_thresholds is None:
        mark_thresholds = []

    data = df[column].dropna()
    label = column

    if log_transform:
        data = np.log(data[data > 0])
        label = f"log({column})"
        if unit:
            label += f" [log$({unit})$]"
    else:
        if unit:
            label += f" [${unit}$]"

    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    fig, ax = plt.subplots(figsize=figsize)

    if nresp and nresp in df.columns and marker_color == 'black':
        mask = df[nresp].loc[data.index] >= 3
        ax.plot(sorted_data[~mask], ecdf[~mask], marker=cdf_marker, linestyle='none',
                color='gray', markersize=marker_size, label='Empirical CDF (nresp < 3)')
        ax.plot(sorted_data[mask], ecdf[mask], marker=cdf_marker, linestyle='none',
                color='black', markersize=marker_size, label='Empirical CDF (nresp ≥ 3)')
    else:
        ax.plot(sorted_data, ecdf, marker=cdf_marker, linestyle='none', label="Empirical CDF", 
                color=marker_color, markersize=marker_size)

    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    best_fit = None
    best_ks_stat = np.inf
    fit_results = {}

    for dist_name in compare_dists:
        dist_obj, dist_str = all_dists.get(dist_name, (None, None))
        if dist_obj is None:
            print(f"⚠️ Distribution '{dist_name}' not recognized. Skipping.")
            continue
        try:
            params = dist_obj.fit(data)
            x = np.linspace(min(sorted_data), max(sorted_data), 1000)
            cdf_theoretical = dist_obj.cdf(x, *params)

            if compute_best_fit:
                ks_stat, _ = stats.kstest(data, dist_str, args=params)
                fit_results[dist_name] = {'params': params, 'ks': ks_stat}

                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_fit = dist_name

                label_line = f"{dist_name} CDF (KS={ks_stat:.3f})"
            else:
                label_line = f"{dist_name} CDF"

            color = next(color_cycle)
            ax.plot(x, cdf_theoretical, label=label_line, linestyle='--', color=color, linewidth=linewidth)
        except Exception as e:
            print(f"❌ Failed to fit and plot '{dist_name}': {e}")

    for threshold in mark_thresholds:
        if log_transform:
            threshold = np.log(threshold)
        if any(sorted_data >= threshold):
            y_val = ecdf[sorted_data >= threshold][0]
            ax.axvline(threshold, color='red', linestyle=':', alpha=1)
            ax.annotate(f"{np.exp(threshold) if log_transform else threshold:.1f}",
                        xy=(threshold, y_val), xytext=(5, 5), textcoords="offset points",
                        fontsize=15, color='red')

    if show_title:
        title = f"Empirical CDF of {label}"
        if compute_best_fit and best_fit:
            title += f" — Best Fit: {best_fit} (KS={best_ks_stat:.3f})"
        ax.set_title(title)

    ax.set_xlabel(x_label if x_label else label)
    ax.set_ylabel(y_label if y_label else "Empirical Cumulative Probability")
    ax.legend(loc=legend_loc)
    #ax.grid(True)

    if xscale == 'log':
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')

    plt.tight_layout()

    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(out_dir / f"{event_id}_{name}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.show()
    return {
        'best_fit': best_fit if compute_best_fit else None,
        'ks_stat': best_ks_stat if compute_best_fit else None,
        'all_fits': fit_results if compute_best_fit else None
    }



# =============================================================================
# Residuals Plot Function
# =============================================================================
def plotdata_residuals(
    df,
    x='distance',
    y='pga',
    log_y=True,
    vs30_col=None,
    fit_func=attenuation_nonlinear_model,
    fit_label='Nonlinear Fit',
    marker='o',
    s=50,
    alpha=0.7,
    color='steelblue',
    figsize=(24, 16),
    dpi=300,
    output_path=None,
    event_id='event',
    name='residuals',
    save_formats=['png'],
    show_title=True,
    unit_y='g',
    unit_x='km',
    legend_loc='best',
    x_ticks=None,
    y_ticks=None,
    x_label=None,
    y_label=None,
    nresp=None, markeredge_width=3, legend_marker_size= 10, legend_marker_edge= 2
):
    """
    Plot residuals between observed and fitted ground motion values as a scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns for x (e.g., distance), y (e.g., PGA), 
        and optionally vs30 or nresp.
    x : str, default='distance'
        Column name for the x-axis variable (typically distance).
    y : str, default='pga'
        Column name for the y-axis variable (typically PGA or spectral acceleration).
    log_y : bool, default=True
        Whether to apply log10 transform to the y data before fitting.
    vs30_col : str, optional
        Column name for Vs30 values for coloring points by soil class.
    fit_func : callable, default=attenuation_nonlinear_model
        Regression model used to fit the attenuation curve.
    fit_label : str, default='Nonlinear Fit'
        Label used for the fitted model in y-axis annotation.
    marker : str, default='o'
        Marker style for the scatter points.
    s : int, default=50
        Marker size.
    alpha : float, default=0.7
        Marker transparency.
    color : str, default='steelblue'
        Default color of scatter points (used if vs30_col and nresp are not provided).
    figsize : tuple, default=(24, 16)
        Size of the figure.
    dpi : int, default=300
        DPI for saving the figure.
    output_path : str or Path, optional
        Path to save the figure.
    event_id : str, default='event'
        Identifier used in saved filenames.
    name : str, default='residuals'
        Output filename prefix.
    save_formats : list of str, default=['png']
        Output formats to save (e.g., ['png', 'pdf']).
    show_title : bool, default=True
        Whether to show the plot title.
    unit_y : str, default='g'
        Unit label for y-axis.
    unit_x : str, default='km'
        Unit label for x-axis.
    legend_loc : str, default='best'
        Location for the plot legend.
    x_ticks : list, optional
        List of custom x-ticks to apply.
    y_ticks : list, optional
        List of custom y-ticks to apply.
    x_label : str, optional
        Override for x-axis label.
    y_label : str, optional
        Override for y-axis label.
    nresp : str, optional
        Column name for number of responses. If provided, points with nresp ≥ 3 are
        outlined in red, otherwise in black. Legend will reflect this distinction.

    Returns
    -------
    dict
        Dictionary containing 'residuals' and 'params' of the fitted model.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Filter required columns
    cols = [x, y]
    if vs30_col: cols.append(vs30_col)
    if nresp: cols.append(nresp)
    data = df[cols].dropna()

    x_vals = data[x].values
    y_vals = data[y].values
    if log_y:
        y_vals = np.log10(np.maximum(y_vals, 1e-10))

    # Fit model and compute residuals
    try:
        popt, _ = curve_fit(fit_func, x_vals, y_vals, p0=[1, -1, 1], maxfev=10000)
        y_pred = fit_func(x_vals, *popt)
        residuals = y_vals - y_pred
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {e}")

    # Assign soil color (marker fill)
    if vs30_col:
        vs30_vals = data[vs30_col].values
        soil_classes, soil_colors = assign_soil_class_and_color(vs30_vals)
        data['soil_class'] = soil_classes
        data['soil_color'] = soil_colors
    else:
        data['soil_color'] = [color] * len(data)

    # Assign edge color based on nresp
    if nresp:
        edge_colors = ['red' if val >= 3 else 'black' for val in data[nresp].values]
    else:
        edge_colors = ['k'] * len(data)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x_vals, residuals,
        c=data['soil_color'], edgecolors=edge_colors,
        s=s, alpha=alpha, marker=marker, linewidth=markeredge_width
    )

    # Reference line
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    # Labels
    xlabel = x_label if x_label else (f"{x} [{unit_x}]" if unit_x else x)
    ylabel = y_label if y_label else f"Residuals [{y}]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_title:
        ax.set_title(f"Residuals vs {x}")

    # Custom ticks
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks])
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # Legend: combine soil and nresp
    legend_elements = []

    if vs30_col:
        soil_class_order = ['A', 'B', 'C', 'D', 'E']
        soil_color_order = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        legend_elements += [
            Line2D([0], [0], marker=marker, linestyle='None',
                   markerfacecolor=fc, markeredgecolor='k',
                   label=f'Soil Class {cls}', markersize=legend_marker_size,markeredgewidth=legend_marker_edge)
            for cls, fc in zip(soil_class_order, soil_color_order)
        ]
    
    if nresp:
        legend_elements += [
            Line2D([0], [0], marker=marker, linestyle='None',
                   markerfacecolor='gray', markeredgecolor='red',
                   label='nresp ≥ 3', markersize=legend_marker_size,markeredgewidth=legend_marker_edge),
            Line2D([0], [0], marker=marker, linestyle='None',
                   markerfacecolor='gray', markeredgecolor='black',
                   label='nresp < 3', markersize=legend_marker_size,markeredgewidth=legend_marker_edge),
        ]

    if legend_elements:
        ax.legend(handles=legend_elements, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    # Final touches
    #ax.grid(True)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()

    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(out_dir / f"{event_id}_{name}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.show()

    return {
        'residuals': residuals,
        'params': popt
    }











# ────────────────────────────────────────────────────────────────────────────
#  HISTOGRAM WITH CONDITIONAL NRESP SPLITTING AND PDF FITS
# ────────────────────────────────────────────────────────────────────────────

def plotdata_histogram(
    df,
    column='pga',
    log_transform=False,
    distributions=None,
    unit=None,
    bins=30,
    output_path=None,
    event_id='event',
    name='histogram',
    save_formats=['png'],
    dpi=300,
    figsize=(24, 16),
    show_title=True,
    linewidth=5,
    alpha=0.6,
    scoring_method='aic',  # options: 'aic', 'bic', 'ks', or None
    legend_loc='best',
    nresp=None,
    x_label=None,
    y_label=None
):
    """
    Plot histogram with fitted probability density functions (PDFs) for a given column.

    If `nresp` is specified, the data is split into two groups (nresp ≥ 3 vs nresp < 3),
    and both histograms and PDFs are shown with different styles.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    column : str
        Name of the column to plot.
    log_transform : bool
        Whether to log-transform the column data.
    distributions : list of str
        Distributions to fit (e.g., 'normal', 'lognorm').
    unit : str
        Unit to show on the x-axis.
    bins : int
        Number of histogram bins.
    output_path : str
        Folder to save the figure (optional).
    event_id : str
        Event identifier for saved file name.
    name : str
        Output file name prefix.
    save_formats : list of str
        File formats to save the plot.
    dpi : int
        Resolution.
    figsize : tuple
        Size of the figure.
    show_title : bool
        Whether to show title.
    linewidth : float
        Line width for PDF curves.
    alpha : float
        Transparency for histogram bars.
    scoring_method : str
        'aic', 'bic', 'ks', or None — used to evaluate best fit.
    legend_loc : str
        Location of the legend.
    nresp : str
        Optional column name for number of responses — splits data.

    Returns
    -------
    dict
        Summary including best-fit distribution, score, and fit parameters.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from pathlib import Path
    import itertools
    from matplotlib.lines import Line2D

    all_dists = {
        'normal': (stats.norm, 'norm'),
        'lognorm': (stats.lognorm, 'lognorm'),
        'gamma': (stats.gamma, 'gamma'),
        'weibull': (stats.weibull_min, 'weibull_min'),
        'expon': (stats.expon, 'expon'),
        'beta': (stats.beta, 'beta'),
        'genextreme': (stats.genextreme, 'genextreme'),
        'pearson3': (stats.pearson3, 'pearson3'),
        'burr': (stats.burr12, 'burr12')
    }

    if distributions is None:
        distributions = list(all_dists.keys())
    dist_map = {name: all_dists[name] for name in distributions if name in all_dists}

    data = df[[column] + ([nresp] if nresp else [])].dropna()
    label = column

    if log_transform:
        data = data[data[column] > 0]
        data[column] = np.log(data[column])
        label = f"log({column})"
        if unit:
            label += f" [log$({unit})$]"
    elif unit:
        label += f" [${unit}$]"

    fig, ax = plt.subplots(figsize=figsize)
    x_vals = np.linspace(data[column].min(), data[column].max(), 1000)
    color_cycle = itertools.cycle(plt.cm.tab10.colors)
    fit_scores = {}
    best_dist = None
    best_score = np.inf

    if nresp:
        high = data[data[nresp] >= 3][column]
        low = data[data[nresp] < 3][column]
        #ax.hist(low, bins=bins, alpha=0.5, density=True, color='lightgray', edgecolor='black', label='nresp < 3')
        #ax.hist(high, bins=bins, alpha=0.5, density=True, color='gray', edgecolor='black', label='nresp ≥ 3')

        ax.hist(low, bins=bins, alpha=0.5, density=True, color='lightgray')
        ax.hist(high, bins=bins, alpha=0.5, density=True, color='gray')
        
        ax.hist(low, bins=bins, density=True, histtype='step', color='black',
                linestyle='--', linewidth=2, label='nresp < 3')
        ax.hist(high, bins=bins, density=True, histtype='step', color='black',
                linestyle='-', linewidth=2, label='nresp ≥ 3')


        for dist_name, (dist_obj, dist_str) in dist_map.items():
            try:
                # < 3
                params_low = dist_obj.fit(low)
                y_low = dist_obj.pdf(x_vals, *params_low)
                ax.plot(x_vals, y_low, linestyle='--', linewidth=linewidth,
                        color=next(color_cycle), label=f"{dist_name} (nresp<3)")
                # ≥ 3
                params_high = dist_obj.fit(high)
                y_high = dist_obj.pdf(x_vals, *params_high)
                ax.plot(x_vals, y_high, linestyle='-', linewidth=linewidth,
                        color=ax.lines[-1].get_color(), label=f"{dist_name} (nresp≥3)")

                # Evaluate best on full data
                params_all = dist_obj.fit(data[column])
                if scoring_method == 'aic':
                    ll = np.sum(dist_obj.logpdf(data[column], *params_all))
                    k = len(params_all)
                    score = 2 * k - 2 * ll
                elif scoring_method == 'bic':
                    ll = np.sum(dist_obj.logpdf(data[column], *params_all))
                    k = len(params_all)
                    score = k * np.log(len(data[column])) - 2 * ll
                elif scoring_method == 'ks':
                    score, _ = stats.kstest(data[column], dist_str, args=params_all)
                else:
                    score = None

                if scoring_method and score is not None:
                    fit_scores[dist_name] = {'params': params_all, 'score': score}
                    if score < best_score:
                        best_score = score
                        best_dist = dist_name
            except Exception as e:
                print(f"❌ Failed to fit and plot '{dist_name}': {e}")
    else:
        # No nresp given → single histogram
        ax.hist(data[column], bins=bins, alpha=alpha, density=True,
                color='lightgray', edgecolor='black', label='Histogram')

        for dist_name, (dist_obj, dist_str) in dist_map.items():
            try:
                params = dist_obj.fit(data[column])
                y = dist_obj.pdf(x_vals, *params)
                color = next(color_cycle)

                if scoring_method == 'aic':
                    ll = np.sum(dist_obj.logpdf(data[column], *params))
                    k = len(params)
                    score = 2 * k - 2 * ll
                elif scoring_method == 'bic':
                    ll = np.sum(dist_obj.logpdf(data[column], *params))
                    k = len(params)
                    score = k * np.log(len(data[column])) - 2 * ll
                elif scoring_method == 'ks':
                    score, _ = stats.kstest(data[column], dist_str, args=params)
                else:
                    score = None

                label_text = f"{dist_name} PDF"
                if scoring_method:
                    label_text += f" ({scoring_method.upper()}={score:.3f})"

                ax.plot(x_vals, y, linestyle='--', color=color, linewidth=linewidth, label=label_text)

                if scoring_method and score < best_score:
                    best_score = score
                    best_dist = dist_name
                if scoring_method:
                    fit_scores[dist_name] = {'params': params, 'score': score}
            except Exception as e:
                print(f"❌ Failed to fit '{dist_name}': {e}")

    if show_title:
        title = f"Histogram with Fitted PDFs of {label}"
        if scoring_method and best_dist:
            title += f"\nBest Fit: {best_dist} ({scoring_method.upper()}={best_score:.3f})"
        ax.set_title(title)


        # Labels
    xlabel = x_label if x_label else label
    ylabel = y_label if y_label else f"Density"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #ax.grid(True)

    if nresp:
        # Clean legend layout
        legend_lines = []
        colors = plt.cm.tab10.colors
        for idx, dist_name in enumerate(dist_map.keys()):
            legend_lines.append(Line2D([0], [0], color=colors[idx % len(colors)], lw=linewidth, label=dist_name))
        legend_lines += [
            Line2D([0], [0], linestyle='--', color='black', lw=linewidth, label='nresp < 3'),
            Line2D([0], [0], linestyle='-', color='black', lw=linewidth, label='nresp ≥ 3')
        ]
        ax.legend(handles=legend_lines, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    plt.tight_layout()

    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(out_dir / f"{event_id}_{name}.{fmt}", dpi=dpi, bbox_inches="tight")

    plt.show()

    return {
        'best_fit': best_dist if scoring_method else None,
        'best_score': best_score if scoring_method else None,
        'scoring_method': scoring_method,
        'all_fits': fit_scores if scoring_method else None
    }




# ==============================================================================
# plotdata_qq_pp
# ==============================================================================
def plotdata_qq_pp(
    df,
    column='pga',
    dist_name=None,
    output_path=None,
    log_if_skewed=True,
    event_id='event',
    name_prefix='qq_pp',
    save_formats=['png'],
    dpi=300,
    figsize=(12, 10),
    show_title=True,
    marker='o',
    marker_color='blue',
    marker_size=16,
    line_color='red',
    line_width=2
):
    """
    Generate QQ and PP plots for a specified column with best-fit or selected distribution.
    Automatically handles log-transform if data is highly skewed.

    Parameters
    ----------
    df : pd.DataFrame
        Data source containing the column to evaluate.
    column : str
        Name of the numeric column to analyze.
    dist_name : str, optional
        Specific distribution to use (e.g., 'lognorm', 'gamma'), or determine best via KS test.
    output_path : str or Path, optional
        Folder path to save the plots.
    log_if_skewed : bool
        Automatically apply log10 if skewness > 1.
    event_id : str
        Earthquake or scenario identifier for filename.
    name_prefix : str
        Prefix for saved image file names.
    save_formats : list of str
        File extensions to save plots (e.g., ['png', 'pdf']).
    dpi : int
        Image resolution.
    figsize : tuple
        Size of the output figure.
    show_title : bool
        Whether to display title on plots.
    marker : str
        Marker style used in scatter points.
    marker_color : str
        Color of the scatter points.
    marker_size : int
        Size of scatter markers.
    line_color : str
        Color of reference line.
    line_width : float
        Width of reference (QQ/PP) line.

    Returns
    -------
    dict
        Dictionary with best-fit distribution name, parameters, and log transformation flag.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy import stats
    from statsmodels.graphics.gofplots import qqplot, ProbPlot

    data = df[column].dropna()
    original_column_name = column

    if data.empty:
        raise ValueError(f"No valid data in column '{column}'")

    skew = stats.skew(data)
    is_skewed = abs(skew) > 1
    transformed = False

    print(f"Skewness of {column}: {skew:.2f} → {'High skew' if is_skewed else 'Acceptable'}")

    if is_skewed and log_if_skewed:
        data = data[data > 0]
        if data.empty:
            raise ValueError(f"Log-transform failed: No positive values in column '{column}'")
        data = np.log(data)
        column += ' [log-transformed]'
        transformed = True
        print("Applied log-transform due to high skewness.")

    all_candidates = {
        'normal': stats.norm,
        'lognorm': stats.lognorm,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min,
        'beta': stats.beta,
        'pearson3': stats.pearson3,
        'burr': stats.burr12,
        'genextreme': stats.genextreme,
    }

    if dist_name is not None:
        if dist_name not in all_candidates:
            raise ValueError(f"'{dist_name}' is not a supported distribution.")
        best_fit_name = dist_name
    else:
        best_fit_name = None
        best_ks = np.inf
        for name, dist in all_candidates.items():
            try:
                params = dist.fit(data)
                stat, _ = stats.kstest(data, name, args=params)
                if stat < best_ks:
                    best_ks = stat
                    best_fit_name = name
            except Exception as e:
                print(f"❌ Failed KS test for {name}: {e}")
        print(f"✅ Best fit based on KS: {best_fit_name}")

    best_dist = all_candidates[best_fit_name]
    best_params = best_dist.fit(data)

    # Separate shape, loc, scale
    n_shapes = len(best_params) - 2 if hasattr(best_dist, 'shapes') and best_dist.shapes else 0
    shape_args = best_params[:n_shapes]
    loc = best_params[n_shapes]
    scale = best_params[n_shapes + 1]

    out_dir = None
    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)

    # QQ Plot
    fig_qq, ax_qq = plt.subplots(figsize=figsize)
    qqplot(data, line=None, dist=best_dist, distargs=shape_args, loc=loc, scale=scale,
           ax=ax_qq, marker=marker, color=marker_color, markersize=marker_size)
    x_vals = np.linspace(min(data), max(data), 100)
    ax_qq.plot(x_vals, x_vals, color=line_color, linestyle='-', linewidth=line_width)

    title_qq = f"QQ Plot of {original_column_name}" + (" [log-transformed]" if transformed else "")
    if show_title:
        ax_qq.set_title(title_qq)
    ax_qq.set_xlabel("Theoretical Quantiles")
    ax_qq.set_ylabel("Sample Quantiles")
    #ax_qq.grid(True)
    plt.tight_layout()
    if out_dir:
        for fmt in save_formats:
            fig_qq.savefig(out_dir / f"{event_id}_{name_prefix}_qq.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.show()

    # PP Plot
    fig_pp, ax_pp = plt.subplots(figsize=figsize)
    probplot = ProbPlot(data, dist=best_dist, distargs=shape_args, loc=loc, scale=scale)
    probplot.ppplot(line=None, ax=ax_pp)
    theoretical_probs = np.linspace(0, 1, len(data))
    ax_pp.plot(theoretical_probs, theoretical_probs, color=line_color, linestyle='-', linewidth=line_width)

    title_pp = f"PP Plot of {original_column_name}" + (" [log-transformed]" if transformed else "")
    if show_title:
        ax_pp.set_title(title_pp)
    ax_pp.set_xlabel("Theoretical Probabilities")
    ax_pp.set_ylabel("Sample Probabilities")
    #ax_pp.grid(True)
    plt.tight_layout()
    if out_dir:
        for fmt in save_formats:
            fig_pp.savefig(out_dir / f"{event_id}_{name_prefix}_pp.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.show()

    return {
        'best_fit': best_fit_name,
        'params': best_params,
        'log_transformed': transformed
    }



# ==============================================================================
# compare Histogram for upto Three dataset
# ==============================================================================



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import cycle
from matplotlib.lines import Line2D
import pandas as pd

def plotdata_compare_histogram(
    datasets,
    column_names,
    log_transform=False,
    distributions=None,
    unit=None,
    bins=30,
    output_path=None,
    event_id='event',
    save_name='compare_histogram',
    save_formats=['png'],
    dpi=300,
    figsize=(24, 16),
    show_title=True,
    linewidth=3,
    alpha=0.5,
    scoring_method='aic',
    legend_loc='best',
    x_label=None,
    y_label=None,
):
    """
    Compare histograms and top-fitted PDFs across multiple datasets.

    Parameters
    ----------
    datasets : list of arrays or Series or single-column DataFrames
        Each dataset to be visualized and compared.
    column_names : list of str
        Labels corresponding to each dataset.
    log_transform : bool
        Whether to apply log10 transformation before plotting/fitting.
    distributions : list of str
        Distribution names to consider for PDF fitting.
    unit : str
        Unit label for x-axis (optional).
    bins : int
        Number of histogram bins.
    output_path : str, optional
        Directory path to save output figure.
    event_id : str
        Earthquake/scenario ID for output naming.
    name : str
        Name prefix for output file.
    save_formats : list of str
        File formats to save (e.g., ['png', 'pdf']).
    dpi : int
        Resolution of the output figure.
    figsize : tuple
        Size of the plot (default: (24,16)).
    show_title : bool
        Whether to display the plot title.
    linewidth : float
        Linewidth of PDF plots.
    alpha : float
        Transparency of histogram bars.
    scoring_method : str
        Scoring method to choose top distributions: 'aic', 'bic', or 'ks'.
    legend_loc : str
        Location of the legend box.
    x_label, y_label : str, optional
        Override axis labels if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot figure.
    fit_summary : dict
        Summary of top 4 distributions for each dataset.
    """
    if not (1 <= len(datasets) <= 4):
        raise ValueError("Function supports only 1 to 4 datasets for clarity. Reduce input size.")

    if len(datasets) == 4:
        print("⚠️ Warning: Plot may be visually crowded with 4 datasets.")

    all_dists = {
        'normal': (stats.norm, 'norm'),
        'lognorm': (stats.lognorm, 'lognorm'),
        'gamma': (stats.gamma, 'gamma'),
        'weibull': (stats.weibull_min, 'weibull_min'),
        'expon': (stats.expon, 'expon'),
        'beta': (stats.beta, 'beta'),
        'genextreme': (stats.genextreme, 'genextreme'),
        'pearson3': (stats.pearson3, 'pearson3'),
        'burr': (stats.burr12, 'burr12')
    }

    if distributions is None:
        distributions = list(all_dists.keys())
    dist_map = {name: all_dists[name] for name in distributions if name in all_dists}

    linestyles = ['-', '--', ':', '-.']
    gray_colors = ['lightgray', 'gray', 'dimgray', 'slategray']
    fig, ax = plt.subplots(figsize=figsize)
    fit_summary = {}
    color_cycle = cycle(plt.cm.tab10.colors)
    dist_color_map = {}

    for idx, raw_data in enumerate(datasets):
        # --- Data prep ---
        if isinstance(raw_data, (list, tuple)):
            data = np.array(raw_data)
        elif isinstance(raw_data, (np.ndarray, pd.Series)):
            data = raw_data.values if hasattr(raw_data, 'values') else raw_data
        elif isinstance(raw_data, pd.DataFrame):
            if raw_data.shape[1] != 1:
                raise ValueError("Each DataFrame dataset must contain exactly one column.")
            data = raw_data.iloc[:, 0].values
        else:
            raise ValueError(f"Unsupported data type for dataset {idx}")

        data = data[~np.isnan(data)].flatten()
        if log_transform:
            data = data[data > 0]
            data = np.log(data)

        if len(data) < 5:
            print(f"⚠️ Dataset '{column_names[idx]}' has too few points for reliable distribution fitting.")

        label = column_names[idx]
        color = gray_colors[idx % len(gray_colors)]
        linestyle = linestyles[idx % len(linestyles)]

        # --- Histogram ---
        ax.hist(data, bins=bins, alpha=alpha, density=True, color=color)
        ax.hist(data, bins=bins, density=True, histtype='step',
                color='black', linestyle=linestyle, linewidth=2)

        # --- Fit distributions ---
        x_vals = np.linspace(np.min(data), np.max(data), 1000)
        dist_scores = {}

        for dist_name, (dist_obj, dist_str) in dist_map.items():
            try:
                params = dist_obj.fit(data)
                y = dist_obj.pdf(x_vals, *params)
                if scoring_method == 'aic':
                    ll = np.sum(dist_obj.logpdf(data, *params))
                    k = len(params)
                    score = 2 * k - 2 * ll
                elif scoring_method == 'bic':
                    ll = np.sum(dist_obj.logpdf(data, *params))
                    k = len(params)
                    score = k * np.log(len(data)) - 2 * ll
                elif scoring_method == 'ks':
                    score, _ = stats.kstest(data, dist_str, args=params)
                else:
                    score = None
                dist_scores[dist_name] = (score, params, y)
            except Exception:
                continue

        top_dists = sorted(dist_scores.items(), key=lambda x: x[1][0])[:4]
        fit_summary[label] = top_dists

        for dist_name, (score, params, y_vals) in top_dists:
            if dist_name not in dist_color_map:
                dist_color_map[dist_name] = next(color_cycle)
            ax.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth,
                    color=dist_color_map[dist_name])

    # Axis and title
    if show_title:
        ax.set_title("Histogram Comparison with Best-Fit Distributions")
    ax.set_xlabel(x_label if x_label else f"Value [{unit}]" if unit else "Value")
    ax.set_ylabel(y_label if y_label else "Density")

    # Custom legend
    legend_elements = []
    for i, name in enumerate(column_names):
        legend_elements.append(Line2D([0], [0], linestyle=linestyles[i % len(linestyles)],
                                      color='black', label=name, linewidth=2))
    for dist_name, color in dist_color_map.items():
        legend_elements.append(Line2D([0], [0], linestyle='-', color=color,
                                      label=dist_name, linewidth=linewidth))
    ax.legend(handles=legend_elements, loc=legend_loc)

    if output_path:
        out_dir = Path(output_path) / "SHAKEtime" / event_id / "seismic_record_assessments"
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_formats:
            fig.savefig(out_dir / f"{event_id}_{save_name}.{fmt}", dpi=dpi, bbox_inches="tight")


    plt.tight_layout()
    return fig, fit_summary
