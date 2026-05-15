import numpy as np
import os
import matplotlib.pyplot as plt
from ..lib import util, plots

colors = ['xkcd:bright blue', 'xkcd:soft green', 'orange', 'purple']


@plots.apply_style
def plot_whitelc(optspec, time, meta, i, fig=None, ax=None):
    '''Plot binned white light curve and indicate
    baseline and in-occultation regions.

    Parameters
    ----------
    optspec : np.ndarray
        The optimally extracted lightcurve(s).
    time : np.ndarray
        The time array.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The occultation number.
    fig : object; optional
        The figure object. Default is None, which creates a new object.
    ax : object; optional
        The axis object. Default is None, which creates a new object.

    Returns
    -------
    fig : object
        The figure object.
    ax : object
        The axis object.
    '''
    toffset = meta.time_offset
    it0, it1, it2, it3, it4, it5 = meta.it

    # Use the pre-computed raw white LC (all integrations, before clipping)
    # so kept and removed points share the same data and normalization.
    # The light curve therefore looks identical regardless of clipping —
    # removed points simply appear in red at their true y-position.
    lc_all_raw = getattr(meta, 's4cal_plot_lc_all', None)
    all_plot_times = getattr(meta, 's4cal_plot_all_times', None)
    kept_mask_plot = getattr(meta, 's4cal_plot_kept_mask',
                             np.ones(len(time), dtype=bool))

    if lc_all_raw is not None and all_plot_times is not None:
        lc_all = lc_all_raw / np.nanmean(lc_all_raw)
        lc_kept = lc_all[kept_mask_plot]
        time_vals = all_plot_times[kept_mask_plot]
        has_removed = bool(np.any(~kept_mask_plot))
        if has_removed:
            lc_removed = lc_all[~kept_mask_plot]
            times_removed = all_plot_times[~kept_mask_plot]
        all_times = all_plot_times
    else:
        # Fallback for old meta objects
        if meta.photometry:
            lc_kept = np.ma.copy(optspec)
        else:
            lc_kept = np.ma.sum(optspec, axis=1)
        lc_kept = np.asarray(lc_kept / np.ma.mean(lc_kept), dtype=float)
        time_vals = np.array(time)
        has_removed = False
        all_times = time_vals

    # Bin both kept and removed data using consistent bin edges
    do_bin = meta.nbin_plot and meta.nbin_plot < len(time_vals)
    if do_bin:
        nbin = meta.nbin_plot
        bin_edges = np.linspace(np.min(all_times), np.max(all_times), nbin + 1)
        kept_idx = np.clip(np.digitize(time_vals, bin_edges) - 1, 0, nbin - 1)
        lc_bin = np.array([
            np.nanmean(lc_kept[kept_idx == b])
            if np.any(kept_idx == b) else np.nan
            for b in range(nbin)
        ])
        time_bin = np.array([
            np.nanmean(time_vals[kept_idx == b])
            if np.any(kept_idx == b) else np.nan
            for b in range(nbin)
        ])
        lc_bin = np.ma.masked_invalid(lc_bin)
        time_bin = np.ma.masked_invalid(time_bin)
        if has_removed:
            rem_idx = np.clip(
                np.digitize(times_removed, bin_edges) - 1, 0, nbin - 1)
            rem_lc_bin = np.array([
                np.nanmean(lc_removed[rem_idx == b])
                if np.any(rem_idx == b) else np.nan
                for b in range(nbin)
            ])
            rem_time_bin = np.array([
                np.nanmean(times_removed[rem_idx == b])
                if np.any(rem_idx == b) else np.nan
                for b in range(nbin)
            ])
            rem_lc_bin = np.ma.masked_invalid(rem_lc_bin)
            rem_time_bin = np.ma.masked_invalid(rem_time_bin)
    else:
        lc_bin = lc_kept
        time_bin = time_vals
        if has_removed:
            rem_lc_bin = lc_removed
            rem_time_bin = times_removed

    if i == 0:
        fig = plt.figure(4202)
        fig.set_size_inches(8, 5, forward=True)
        fig.clf()
        ax = fig.subplots(1, 1)
        # Grey unbinned background (kept integrations)
        ax.plot(time_vals - toffset, lc_kept, '.', color='0.5', alpha=0.2,
                ms=2, zorder=1, rasterized=True)
        # Faint red unbinned clipped integrations
        if has_removed:
            ax.plot(times_removed - toffset, lc_removed, '.', color='red',
                    alpha=0.2, ms=2, zorder=2, rasterized=True)
        # Binned kept data on top
        ax.plot(time_bin - toffset, lc_bin, '.', color='0.2', alpha=0.8,
                zorder=3, label='Binned White LC')
        # Binned removed data in red
        if has_removed:
            ax.plot(rem_time_bin - toffset, rem_lc_bin, '.', color='red',
                    alpha=0.8, zorder=3, label='Manually Clipped')
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx((ymin, ymax), time[it0]-toffset, time[it1]-toffset,
                     color=colors[1], alpha=0.2)
    ax.fill_betweenx((ymin, ymax), time[it4]-toffset, time[it5]-toffset,
                     color=colors[1], alpha=0.2)
    ax.fill_betweenx((ymin, ymax), time[it2]-toffset, time[it3]-toffset,
                     color=colors[0], alpha=0.2)
    ax.vlines([time[it1]-toffset, time[it4]-toffset,
              time[it0]-toffset, time[it5]-toffset],
              ymin, ymax, color=colors[1], label='Baseline Regions')
    ax.vlines([time[it2]-toffset, time[it3]-toffset],
              ymin, ymax, color=colors[0], label='In-Occultation Region')
    if i == 0:
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='best')
        ax.set_xlabel(f"Time ({time.time_units})")
        ax.set_ylabel("Normalized Flux")
    fname = 'figs'+os.sep+'fig4202_WhiteLC'
    fig.savefig(meta.outputdir+fname+plots.get_filetype(),
                bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
    return fig, ax


@plots.apply_style
def plot_stellarSpec(meta, ds):
    '''Plot calibrated stellar spectra from
    baseline and in-occultation regions.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    ds : Xarray DataSet
        The DataSet object containing the extracted flux values.
    '''
    if meta.s4cal_plotErrorType == 'stderr':
        # Use the standard error of the mean
        base_err = ds.base_ferr
        ecl_err = ds.ecl_ferr
    elif meta.s4cal_plotErrorType == 'stddev':
        # Use the standard deviation
        base_err = ds.base_fstd
        ecl_err = ds.ecl_fstd
    else:
        raise ValueError(f"Unknown error type: {meta.s4cal_plotErrorType}")

    fig = plt.figure(4201)
    fig.set_size_inches(8, 5, forward=True)
    fig.clf()
    ax = fig.subplots(1, 1)
    for i in range(len(ds.time)):
        ax.errorbar(ds.wavelength, ds.base_flux[:, i], base_err[:, i],
                    fmt='.', capsize=2, ms=2, color=colors[1],
                    label=f'Baseline ({ds.time.values[i]})')
        ax.errorbar(ds.wavelength, ds.ecl_flux[:, i], ecl_err[:, i],
                    fmt='.', capsize=2, ms=2, color=colors[0],
                    label=f'In-Occultation ({ds.time.values[i]})')

    ax.legend(loc='best')
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(f"Flux ({ds.base_flux.flux_units})")

    fname = 'figs'+os.sep+'fig4201_CalStellarSpec'
    fig.savefig(meta.outputdir+fname+plots.get_filetype(),
                bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
    return
