import numpy as np
from scipy.optimize import minimize
from . import imageedit as ie
from . import gaussian as g
from . import gaussian_min as gmin
from ..S3_data_reduction import plots_s3


def centerdriver(method, data, meta, i=None, m=None):
    """
    Use the center method to find the center of a star, starting
    from position guess.

    Parameters
    ----------
    method : string
        Name of the centering method to use.
    data : Xarray Dataset
        The Dataset object in which the centroid data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int; optional
        The current integration. Defaults to None.
    m : int; optional
        The file number. Defaults to None.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the centroid data stored inside.
    """
    # Apply the mask
    mask = data.mask.values[i]
    flux = np.ma.masked_where(mask, data.flux.values[i])
    err = np.ma.masked_where(mask, data.err.values[i])
    saved_ref_median_frame = data.medflux.values

    yxguess = [data.centroid_y.values[i], data.centroid_x.values[i]]

    if method[-4:] == '_sec':
        trim = meta.ctr_cutout_size
    else:
        trim = 0

    if method in ['fgc_pri', 'fgc_sec']:
        # Trim the image if requested
        if trim != 0:
            # Integer part of center
            cen = np.rint(yxguess)
            # Center in the trimed image
            loc = (trim, trim)
            # Do the trim:
            flux, mask, err = ie.trimimage(flux, cen, loc, mask=mask, uncd=err)
        else:
            cen = np.array([0, 0])
            loc = np.rint(yxguess)
        weights = 1.0 / np.abs(err)
    else:
        trim = 0
        loc = yxguess
        cen = np.array([0, 0])
        # Subtract median BG because photutils sometimes has a hard time
        # fitting for a constant offset
        flux -= np.ma.median(flux)

    # If all data is bad:
    if np.all(mask):
        raise Exception('Bad Frame Exception!')

    # Get the center with one of the methods:
    if method in ['fgc_pri', 'fgc_sec']:
        sy, sx, y, x = g.fitgaussian(flux, yxguess=loc, mask=mask,
                                     weights=weights,
                                     fitbg=1, maskg=False)[0][0:4]
    elif method == 'mgmc_pri':
        # Median frame creation + first centroid
        # HDL edit commented out below
        # x, y, refrence_median_frame = gmin.pri_cent(img, msk, meta,
        #                                            saved_ref_median_frame)
        
        x, y = gmin.pri_cent(flux, mask, meta, saved_ref_median_frame)
        sy, sx = np.nan, np.nan
        
    elif method == 'mgmc_sec':
        # Second enhanced centroid position + gaussian widths
        sy, sx, y, x = gmin.mingauss(flux, mask, yxguess=loc, meta=meta)

    # only plot when we do the second fit
    if (meta.isplots_S3 >= 3 and method[-4:] == '_sec' and i < meta.nplots):
        plots_s3.phot_centroid_fgc(flux, mask, x, y, sx, sy, i, m, meta)

    # Make trimming correction, then store centroid positions and
    # the Gaussian 1-sigma half-widths
    data.centroid_x.values[i] = x + cen[1] - trim
    data.centroid_y.values[i] = y + cen[0] - trim
    data.centroid_sx.values[i] = sx
    data.centroid_sy.values[i] = sy

    return data

def xypos(data, meta, i=None, m=None):
    '''Finds and records the stellar movement of the stellar centroid
    relative to a central point. Often we wee that the centroid moves
    in a direction over time and is not exactly stable.

    The xypos is essentially the xpos or ypos but projected onto the axis
    defined by how the centroid moves on the detector during the time series.

    Parameters
    ----------
    data : some kind of xarray that contains x- and y-centroid values

    Returns
    -------
    data : xypos
        The updated frame parameters array.  Contains the xypos for each time.

    Notes
    -----
    History:

    - 2025-08-01, Hannah Diamond-Lowe
        Trying this out, not sure if it's useful.
    '''    

    # Apply the mask
    mask = data.mask.values[i]
    flux = np.ma.masked_where(mask, data.flux.values[i])
    err = np.ma.masked_where(mask, data.err.values[i])
    saved_ref_median_frame = data.medflux.values
    
    xpos = data.centroid_x.values
    ypos = data.centroid_y.values
    
    # Sort
    idx = np.argsort(xpos)
    x = xpos[idx]
    y = ypos[idx]
    
    # make a super simple covariance matrix
    yerr = np.full_like(ypos, 0.00005)
    cov = np.diag(yerr**2)
    
    # Pre-compute inverse and log-determinant
    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    
    # NLL function
    def nll(params):
        m, b = params
        model = m * x + b  # use unscaled x
        res = y - model
        return 0.5 * res @ inv_cov @ res + 0.5 * logdet
    
    # Fit
    result = minimize(nll, x0=[0.8, 0.0], method='L-BFGS-B')
    m, b = result.x[0], result.x[1]
    #print(m, b)

    #get the midpoint
    x_mid = np.mean(x)
    y_mid = m * x_mid + b  # project onto the line
    
    # Perpendicular slope
    if np.isclose(m, 0):
        # horizontal line → perpendicular is vertical
        m_perp = np.inf
    else:
        m_perp = -1 / m
        b_perp = y_mid - m_perp * x_mid

    print(m_perp, b_perp)
    
    # Plot perpendicular line
    if np.isfinite(m_perp):
        x_perp = np.linspace(x_mid - 0.025, x_mid + 0.025, 100)
        y_perp = m_perp * (x_perp - x_mid) + y_mid
    else:
        # Vertical line case
        pass

    def swap_fit_and_perp(m, b, m_perp, b_perp, threshold=0):
        """
        If m > threshold, swap the fit line and perpendicular line parameters.
        Returns: (m_new, b_new, m_perp_new, b_perp_new)
        """
        if m > threshold:
            return m_perp, b_perp, m, b
        else:
            return m, b, m_perp, b_perp

    # Perpendicular line through (x_mid, y_mid), slope m_perp
    x0, y0 = x_mid, y_mid
    
    # if the fit line 
    if m>0: m, b, m_perp, b_perp = swap_fit_and_perp(m, b, m_perp, b_perp)
    
    def signed_normal_distance_to_line(x, y, m_line, b_line):
        """
        Signed shortest distance from point (x, y) to line y = m_line * x + b_line.
        Positive if point lies above the line, negative if below.
        """
        denom = -np.sqrt(1 + m_line**2)
        return (m_line * x - y + b_line) / denom

    def signed_distance_to_midpoint(x, y, x_mid, y_mid, m_perp):
        """
        Compute signed Euclidean distance from each (x, y) point to (x_mid, y_mid),
        with the sign determined by position relative to the perpendicular line 
        with slope m_perp passing through the midpoint.
        """
        # Euclidean distance to midpoint
        dx = x - x_mid
        dy = y - y_mid
        distances = np.sqrt(dx**2 + dy**2)
   
        # Determine side of line: use line equation y - y_mid = m_perp * (x - x_mid)
        # Positive if point lies above the line, negative if below
        side = np.sign(dy - m_perp * dx)
    
        return distances * side
    
    signed_dist_to_mid  = signed_distance_to_midpoint(x, y, x_mid, y_mid, m_perp)

    distances_to_mid_order = np.empty_like(signed_dist_to_mid)
    distances_to_mid_order[idx] = signed_dist_to_mid

    xypos = distances_to_mid_order

    return xypos