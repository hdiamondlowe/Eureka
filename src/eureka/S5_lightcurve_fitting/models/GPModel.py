import numpy as np
import george
from george import kernels
import celerite2

from .Model import Model
from ..likelihood import update_uncertainty
from ...lib.split_channels import split

try:
    import tinygp
except ModuleNotFoundError:
    # tinygp is optional and isn't supported yet, so don't throw an exception
    # if it isn't installed
    tinygp = None


# Names that overlap between GP kernel inputs and linear
# decorrelation models (CentroidModel).  Used to prevent the same
# covariate being used in both places.
GP_CENTROID_NAMES = {'xpos', 'ypos', 'xwidth', 'ywidth', 'xy_pos'}


class GPModel(Model):
    """Model for Gaussian Process (GP)"""
    def __init__(self, kernel_types, kernel_input_names, lc,
                 gp_code_name='celerite', normalize=False,
                 useHODLR=False, gp_subsample=None, **kwargs):
        """Initialize the GP model.

        Parameters
        ----------
        kernel_types : list[str]
            The types of GP kernels to use (e.g., ['Matern32']).
        kernel_input_names : list[str]
            Names of GP inputs.  Supported names are 'time', 'xpos',
            'ypos', 'xwidth', and 'ywidth'.  When more than one name
            is given together with george or tinygp, each kernel
            operates on the corresponding input dimension and the
            kernels are summed.
        lc : eureka.S5_lightcurve_fitting.lightcurve
            The current lightcurve object.
        gp_code_name : {'george','celerite','tinygp'}; optional
            GP backend. Default is 'celerite'.
        normalize : bool; optional
            If True, standardize inputs (mean 0, std 1). Default False.
            For details on the benefits of normalization, see e.g.
            Evans et al. 2017.
        useHODLR : bool; optional
            If True and gp_code_name == 'george', use george's HODLRSolver.
            Default is False.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Centroid vectors (xpos, ypos, xwidth, ywidth) should be
            passed here when the corresponding name appears in
            kernel_input_names.
        """
        super().__init__(kernel_types=kernel_types,
                         nkernels=len(kernel_types),
                         kernel_input_names=kernel_input_names,
                         kernel_inputs=None,
                         gp_code_name=gp_code_name, normalize=normalize,
                         useHODLR=useHODLR, fit_lc=np.ma.ones(lc.flux.shape),
                         flux=lc.flux, unc=lc.unc, unc_fit=lc.unc_fit,
                         gp_subsample=gp_subsample,
                         **kwargs)
        self.name = 'GP'

        # Subsampling factor for tinygp (None or 1 means no subsampling)
        self.gp_subsample = gp_subsample if gp_subsample is not None else 1

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Do some initial sanity checks and raise errors if needed
        if self.gp_code_name == 'celerite':
            if self.nkernels > 1:
                raise AssertionError(
                    'celerite2 cannot compute multi-dimensional GPs. '
                    'Use a single kernel or a different GP backend.'
                )
            if self.kernel_types[0] != 'Matern32':
                raise AssertionError(
                    'Our celerite2 implementation currently supports only '
                    'a Matern32 kernel.'
                )
            non_time = [n for n in kernel_input_names if n != 'time']
            if non_time:
                raise AssertionError(
                    'celerite2 only supports 1-D (time) GP inputs. '
                    f'Non-time inputs requested: {non_time}. '
                    'Use GP_package = tinygp or george for multi-'
                    'dimensional GP inputs.'
                )

    def update(self, newparams, **kwargs):
        """Update parameters and refresh uncertainty fit array."""
        # Inherit from Model class
        super().update(newparams, **kwargs)
        self.unc_fit = update_uncertainty(newparams, self.nints, self.unc,
                                          self.freenames, self.nchannel_fitted)

    def eval(self, fit_lc, channel=None, gp=None, **kwargs):
        """Compute the GP predictive mean of residuals.

        Parameters
        ----------
        fit_lc : ndarray
            The current (non-GP) model evaluation.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite2.GP, george.GP, or tinygp.GaussianProcess; optional
            Pre-built GP object to reuse; if None, a new one is created.
        **kwargs : dict
            Must include 'time' if self.time is None.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            Predicted GP systematics (same shape as time).
        """
        input_gp = gp
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.ma.array([])
        for chan in channels:
            if self.nchannel_fitted > 1:
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if nchan > 1:
                    fit_temp = split([fit_lc, ], self.nints, chan)[0]
                else:
                    # If only a specific channel is being evaluated, then only
                    # that channel's fitted model will be passed in
                    fit_temp = fit_lc
            else:
                # get flux and uncertainties for current channel
                flux = self.flux
                fit_temp = fit_lc
                unc_fit = self.unc_fit
            residuals = np.ma.masked_invalid(flux-fit_temp)
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time
            residuals = np.ma.masked_where(time.mask, residuals)

            # Remove poorly handled masked values
            good = ~np.ma.getmaskarray(residuals)

            if self.gp_code_name == 'george':
                unc_good = unc_fit[good]
                res_good = residuals[good]
                if input_gp is None:
                    gp = self.setup_GP(chan, good=good)
                else:
                    gp = input_gp
                    if self.kernel_inputs is None:
                        self.setup_inputs()
                kin = self.kernel_inputs[chan][:, good].T
                # george requires plain ndarrays, not masked arrays
                if isinstance(kin, np.ma.MaskedArray):
                    kin = np.asarray(kin.data)
                if isinstance(unc_good, np.ma.MaskedArray):
                    unc_good = np.asarray(unc_good.data)
                res_good = np.asarray(res_good)
                gp.compute(kin, unc_good)
                mu = gp.predict(res_good, kin, return_cov=False)
            elif self.gp_code_name == 'celerite':
                unc_good = unc_fit[good]
                res_good = residuals[good]
                if input_gp is None:
                    gp = self.setup_GP(chan, good=good)
                else:
                    gp = input_gp
                    if self.kernel_inputs is None:
                        self.setup_inputs()
                kin = self.kernel_inputs[chan][0][good]
                gp.compute(kin, yerr=unc_good)
                mu = gp.predict(res_good)
            elif self.gp_code_name == 'tinygp':
                if tinygp is None:
                    raise RuntimeError('tinygp is not available.')
                # For tinygp/JAX, also exclude positions where kernel
                # inputs are masked (JAX cannot handle masked arrays)
                if self.kernel_inputs is None:
                    self.setup_inputs()
                kin_mask = np.ma.getmaskarray(
                    self.kernel_inputs[chan]).any(axis=0)
                good &= ~kin_mask
                unc_good = unc_fit[good]
                res_good = residuals[good]
                if input_gp is None:
                    gp = self.setup_GP(
                        chan, good=good,
                        diag=np.asarray(unc_good)**2)
                else:
                    gp = input_gp
                cond_gp = gp.condition(np.asarray(res_good)).gp
                mu = cond_gp.loc
            else:
                raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

            # Re-insert and mask bad values
            mu_full = np.ma.zeros(len(time))
            mu_full[good] = mu
            mu_full = np.ma.masked_where(~good, mu_full)

            # Append to the full list
            lcfinal = np.ma.append(lcfinal, mu_full)

        return lcfinal

    def eval_per_kernel(self, fit_lc, channel=None, **kwargs):
        """Decompose the GP prediction into per-kernel contributions.

        Each kernel component's contribution is computed as
        ``K_k @ alpha`` where ``alpha = (K_total + sigma^2 I)^{-1} y``
        and ``K_k`` is the covariance matrix from kernel *k* alone.
        The sum of all components equals the total GP prediction from
        :meth:`eval`.

        Parameters
        ----------
        fit_lc : ndarray
            The current (non-GP) model evaluation.
        channel : int; optional
            If not None, only consider one of the channels.
        **kwargs : dict
            Must include 'time' if self.time is None.

        Returns
        -------
        components : dict
            ``{kernel_input_name: np.ma.MaskedArray}`` with one entry
            per kernel dimension.  Each array has the same shape as
            the result of :meth:`eval`.
        """
        nchan, channels = self._channels(channel)

        if self.time is None:
            self.time = kwargs.get('time')

        # Initialise per-kernel result arrays
        results = {name: np.ma.array([])
                   for name in self.kernel_input_names}

        for chan in channels:
            if self.nchannel_fitted > 1:
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if nchan > 1:
                    fit_temp = split([fit_lc, ], self.nints, chan)[0]
                else:
                    fit_temp = fit_lc
            else:
                flux = self.flux
                fit_temp = fit_lc
                unc_fit = self.unc_fit

            residuals = np.ma.masked_invalid(flux - fit_temp)
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time
            residuals = np.ma.masked_where(time.mask, residuals)
            good = ~np.ma.getmaskarray(residuals)

            # ----------------------------------------------------------
            # Backend-specific: solve for alpha then decompose K_k*alpha
            # ----------------------------------------------------------
            if self.gp_code_name == 'george':
                if self.kernel_inputs is None:
                    self.setup_inputs()
                unc_good = unc_fit[good]
                res_good = residuals[good]
                kin = self.kernel_inputs[chan][:, good].T
                if isinstance(kin, np.ma.MaskedArray):
                    kin = np.asarray(kin.data)
                if isinstance(unc_good, np.ma.MaskedArray):
                    unc_good = np.asarray(unc_good.data)
                res_good = np.asarray(res_good)

                gp = self.setup_GP(chan, good=good)
                gp.compute(kin, unc_good)
                alpha = gp.solver.apply_inverse(res_good)

                for k in range(self.nkernels):
                    single_k = self.get_kernel(self.kernel_types[k], k, chan)
                    K_k = single_k.get_value(kin)
                    mu_k = K_k @ alpha
                    mu_full = np.ma.zeros(len(time))
                    mu_full[good] = mu_k
                    mu_full = np.ma.masked_where(~good, mu_full)
                    name = self.kernel_input_names[k]
                    results[name] = np.ma.append(results[name], mu_full)

            elif self.gp_code_name == 'celerite':
                # Single kernel — the one component equals the total
                unc_good = unc_fit[good]
                res_good = residuals[good]
                if self.kernel_inputs is None:
                    self.setup_inputs()
                kin = self.kernel_inputs[chan][0][good]
                gp = self.setup_GP(chan, good=good)
                gp.compute(kin, yerr=unc_good)
                mu = gp.predict(res_good)
                mu_full = np.ma.zeros(len(time))
                mu_full[good] = mu
                mu_full = np.ma.masked_where(~good, mu_full)
                name = self.kernel_input_names[0]
                results[name] = np.ma.append(results[name], mu_full)

            elif self.gp_code_name == 'tinygp':
                if tinygp is None:
                    raise RuntimeError('tinygp is not available.')
                import jax
                import jax.numpy as jnp

                if self.kernel_inputs is None:
                    self.setup_inputs()
                kin_mask = np.ma.getmaskarray(
                    self.kernel_inputs[chan]).any(axis=0)
                good &= ~kin_mask

                unc_good = unc_fit[good]
                res_good = residuals[good]
                gp = self.setup_GP(
                    chan, good=good,
                    diag=np.asarray(unc_good)**2)

                kin = self.kernel_inputs[chan][:, good].T
                if kin.ndim == 2 and kin.shape[1] == 1:
                    kin = kin[:, 0]
                if isinstance(kin, np.ma.MaskedArray):
                    kin = np.asarray(kin.data)
                kin_jax = jnp.array(kin)
                res_jax = jnp.array(np.asarray(res_good))
                unc_jax = jnp.array(np.asarray(unc_good))

                # Build full covariance and solve for alpha.
                # Observation noise (unc^2) is now included at
                # construction time via setup_GP(diag=unc^2), so
                # replicate that here.
                K_full = gp.kernel(kin_jax, kin_jax)
                noise = jnp.diag(unc_jax**2)
                alpha = jnp.linalg.solve(K_full + noise, res_jax)

                for k in range(self.nkernels):
                    single_k = self.get_kernel(self.kernel_types[k], k, chan)
                    K_k = single_k(kin_jax, kin_jax)
                    mu_k = np.array(K_k @ alpha)
                    mu_full = np.ma.zeros(len(time))
                    mu_full[good] = mu_k
                    mu_full = np.ma.masked_where(~good, mu_full)
                    name = self.kernel_input_names[k]
                    results[name] = np.ma.append(results[name], mu_full)
            else:
                raise ValueError(
                    f'Unknown gp_code_name: {self.gp_code_name}')

        return results

    def setup_inputs(self):
        """Build kernel input arrays; standardize if requested.

        Supported input names are 'time', 'xpos', 'ypos', 'xwidth',
        and 'ywidth'.  When normalize=True, inputs are standardized to
        zero mean and unit standard deviation.
        """
        # Map from input name to the corresponding stored vector.
        _input_vectors = {
            'time': self.time,
            'xpos': getattr(self, 'xpos', None),
            'ypos': getattr(self, 'ypos', None),
            'xwidth': getattr(self, 'xwidth', None),
            'ywidth': getattr(self, 'ywidth', None),
            'xy_pos': getattr(self, 'xy_pos', None),
        }

        # Store by real channel id to avoid index mismatches.
        self.kernel_inputs = {}
        for chan in self.fitted_channels:
            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            kin_chan = np.ma.zeros((0, time.size))
            for name in self.kernel_input_names:
                vec = _input_vectors.get(name)
                if vec is None:
                    raise ValueError(
                        f"GP kernel input '{name}' is not available. "
                        f"Supported inputs: {list(_input_vectors.keys())}. "
                        "Make sure the corresponding vector is passed to "
                        "GPModel (e.g. via the ECF kernel_inputs setting)."
                    )
                x = np.ma.copy(vec)

                if self.multwhite:
                    x = split([x, ], self.nints, chan)[0]

                if self.normalize:
                    x = (x-np.ma.mean(x))/np.ma.std(x)

                kin_chan = np.ma.append(kin_chan, x[np.newaxis], axis=0)

            self.kernel_inputs[chan] = kin_chan

    def gp_param_suggestions(self):
        """Compute suggested GP parameter bounds from the kernel inputs.

        Returns a list of human-readable strings with suggestions for
        minimum/maximum length scale (``m``) and amplitude (``A``) for
        each kernel, based on the spacing and range of the actual kernel
        input data.  Takes ``gp_subsample`` into account when computing
        the effective point spacing.

        This method must be called after ``setup_inputs()`` has populated
        ``self.kernel_inputs``.

        Returns
        -------
        list of str
            One string per kernel with suggested parameter bounds.
        """
        if self.kernel_inputs is None:
            self.setup_inputs()

        lines = []
        lines.append("GP parameter suggestions (based on kernel input data):")

        # Use the first fitted channel for diagnostics
        chan = self.fitted_channels[0]
        kin = self.kernel_inputs[chan]  # shape (nkernels, N)

        for k in range(self.nkernels):
            ki = '' if k == 0 else str(k)
            name = self.kernel_input_names[k]

            x = np.ma.compressed(kin[k])  # drop masked values
            N_full = len(x)

            # When subsampling, compute spacing from the actual subsampled
            # points (every sub-th integration in time order) rather than
            # multiplying the full-resolution spacing.
            sub = self.gp_subsample
            if sub > 1:
                x_sub = x[::sub]
            else:
                x_sub = x
            N_eff = len(x_sub)

            x_sub_sorted = np.sort(x_sub)
            dx_sub = np.diff(x_sub_sorted)
            dx_sub_pos = dx_sub[dx_sub > 0]

            if len(dx_sub_pos) == 0:
                lines.append(
                    f"  kernel {k} ({name}): all input values identical "
                    f"— cannot compute suggestions")
                continue

            min_spacing = np.min(dx_sub_pos)
            med_spacing = np.median(dx_sub_pos)
            total_range = x_sub_sorted[-1] - x_sub_sorted[0]

            # Length-scale suggestions (in log space, as m = log(ℓ))
            log_ell_min = np.log(min_spacing)
            log_ell_max = np.log(total_range)

            # Amplitude: estimate the flux variance explained by *this*
            # kernel input by sorting the (subsampled) flux by the covariate
            # value, binning it, and computing the variance of the bin means.
            # This isolates how much the flux changes as a function of each
            # input independently, giving a per-kernel amplitude estimate
            # rather than the same total flux scatter for all kernels.
            flux_full = np.ma.compressed(self.flux)
            flux_sub = flux_full[::sub] if sub > 1 else flux_full
            # Trim to same length as x_sub in case flux is longer
            flux_sub = flux_sub[:N_eff]
            # Sort both flux and covariate by covariate value
            sort_idx = np.argsort(x_sub[:N_eff])
            flux_sorted = flux_sub[sort_idx]
            # Use ~10 bins (at least 5) to estimate covariate-explained variance
            n_bins = max(5, N_eff // 10)
            bins = np.array_split(flux_sorted, n_bins)
            bin_means = np.array([b.mean() for b in bins if len(b) > 0])
            amp_estimate = float(np.var(bin_means))

            if amp_estimate > 0:
                log_A_upper = np.log(amp_estimate)
            else:
                # Fallback: total flux variance
                log_A_upper = np.log(np.var(flux_sub)) if np.var(flux_sub) > 0 else 0.0
            # Lower bound: 4 orders of magnitude below the upper estimate
            log_A_lower = log_A_upper - 9.2  # log(1e-4) ≈ -9.2

            lines.append(f"  kernel {k} '{name}' (param: A{ki}, m{ki}):")
            if sub > 1:
                lines.append(
                    f"    gp_subsample = {sub} → {N_full} pts → "
                    f"{N_eff} effective")
            else:
                lines.append(
                    f"    N = {N_full}")
            if self.normalize:
                lines.append(
                    f"    (inputs are normalized to zero mean, unit std)")

            lines.append(
                f"    input range:   {x_sub_sorted[0]:.6g} to "
                f"{x_sub_sorted[-1]:.6g}"
                f"  (total span = {total_range:.6g})")
            lines.append(
                f"    point spacing (subsampled): min = {min_spacing:.6g}, "
                f"median = {med_spacing:.6g}")
            lines.append(
                f"    suggested m{ki} (log length-scale) range: "
                f"[{log_ell_min:.2f}, {log_ell_max:.2f}]")
            lines.append(
                f"      ℓ_min = {np.exp(log_ell_min):.4g} "
                f"(≈ min spacing; shorter will overfit)")
            lines.append(
                f"      ℓ_max = {np.exp(log_ell_max):.4g} "
                f"(≈ total range; longer will underfit)")
            lines.append(
                f"    suggested A{ki} (log amplitude) range: "
                f"[{log_A_lower:.2f}, {log_A_upper:.2f}]")
            lines.append(
                f"      (flux variance explained by '{name}': "
                f"σ_bins = {np.sqrt(amp_estimate):.4g}; "
                f"based on bin-mean variance — adjust using residuals)")

        return lines

    def setup_GP(self, chan=0, good=None, diag=None):
        """Construct the GP object for channel index c.

        Parameters
        ----------
        chan : int; optional
            The current channel index. Defaults to 0.
        good : ndarray of bool; optional
            Mask of valid data points.  Required for tinygp (which
            needs the training coordinates at construction time).
            Ignored by george and celerite backends.  Defaults to None
            (use all points).
        diag : ndarray; optional
            Observation noise variance (sigma^2) for each good data
            point.  Required for tinygp so that observation noise is
            included in the training covariance; ignored by george
            and celerite (which add noise via their compute() calls).

        Returns
        -------
        celerite2.GaussianProcess, george.GP, or tinygp.GaussianProcess
            The GP instance for the requested backend.
        """
        if self.kernel_inputs is None:
            self.setup_inputs()

        # Build kernel as a sum over per-kernel components
        kernel = self.get_kernel(self.kernel_types[0], 0, chan)
        for k in range(1, self.nkernels):
            kernel += self.get_kernel(self.kernel_types[k], k, chan)

        # Make the gp object
        if self.gp_code_name == 'george':
            if self.useHODLR:
                solver = george.solvers.HODLRSolver
            else:
                solver = None
            gp = george.GP(kernel, mean=0, fit_mean=False, solver=solver)
        elif self.gp_code_name == 'celerite':
            gp = celerite2.GaussianProcess(kernel, mean=0, fit_mean=False)
        elif self.gp_code_name == 'tinygp':
            if tinygp is None:
                raise RuntimeError('tinygp is not available.')
            kin = self.kernel_inputs[chan][:, good].T if good is not None \
                else self.kernel_inputs[chan].T  # shape (ngood, nkernels)
            # For 1-D input, squeeze to (ngood,) for simplicity
            if kin.ndim == 2 and kin.shape[1] == 1:
                kin = kin[:, 0]
            # JAX does not support masked arrays; convert to plain
            # ndarray (masked positions already excluded by good mask)
            if isinstance(kin, np.ma.MaskedArray):
                kin = np.asarray(kin.data)
            gp = tinygp.GaussianProcess(kernel, kin, mean=0.0,
                                        diag=diag)
        else:
            raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return gp

    def get_kernel(self, kernel_name, k, chan=0):
        """Return a backend-specific kernel instance.

        Parameters
        ----------
        kernel_name : str
            Kernel type ('Matern32', 'ExpSquared', 'RationalQuadratic',
            'Exp').
        k : int
            Kernel index (0-based).
        chan : int; optional
            The current channel index. Defaults to 0.

        Returns
        -------
        kernel
            The requested kernel.

        Raises
        ------
        AssertionError
            When an unsupported kernel/backend combination is requested.
        """
        # Read per-kernel, per-channel params on demand using suffix rules.
        # A{ki}, m{ki} where ki = '' for k==0 else '1','2',...
        ki = '' if k == 0 else str(k)
        amp_log = self._get_param_value(f'A{ki}', chan=chan)
        metric_log = self._get_param_value(f'm{ki}', chan=chan)

        if self.gp_code_name == 'george':
            amp = np.exp(amp_log)
            metric = np.exp(metric_log*2)

            if kernel_name == 'Matern32':
                kernel = amp*kernels.Matern32Kernel(
                    metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'ExpSquared':
                kernel = amp*kernels.ExpSquaredKernel(
                    metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'RationalQuadratic':
                kernel = amp*kernels.RationalQuadraticKernel(
                    log_alpha=0, metric=metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'Exp':
                kernel = amp*kernels.ExpKernel(
                    metric, ndim=self.nkernels, axes=k)
            else:
                raise AssertionError(
                    f'Unsupported kernel for george: {kernel_name}. '
                    'Supported: Matern32, ExpSquared, RationalQuadratic, Exp.'
                )
        elif self.gp_code_name == 'celerite':
            # celerite2: Matern32 term with sigma, rho
            sigma = np.sqrt(np.exp(amp_log))
            rho = np.exp(metric_log)
            if kernel_name != 'Matern32':
                raise AssertionError('celerite2 path only supports Matern32,'
                                     f' got {kernel_name}.')
            kernel = celerite2.terms.Matern32Term(sigma=sigma, rho=rho)
        elif self.gp_code_name == 'tinygp':
            if tinygp is None:
                raise RuntimeError('tinygp is not available.')
            amp = np.exp(amp_log)
            metric = np.exp(metric_log)  # tinygp takes scale (length-scale), not metric (length-scale²)

            if kernel_name == 'Matern32':
                base_kernel = amp*tinygp.kernels.Matern32(metric)
            elif kernel_name == 'ExpSquared':
                base_kernel = amp*tinygp.kernels.ExpSquared(metric)
            elif kernel_name == 'RationalQuadratic':
                base_kernel = amp*tinygp.kernels.RationalQuadratic(
                    alpha=1, scale=metric)
            elif kernel_name == 'Exp':
                base_kernel = amp*tinygp.kernels.Exp(metric)
            else:
                raise AssertionError(
                    f'Unsupported kernel for tinygp: {kernel_name}. '
                    'Supported: Matern32, ExpSquared, '
                    'RationalQuadratic, Exp.')
            # When using multiple input dimensions, restrict each
            # kernel to its corresponding axis via Subspace.
            if self.nkernels > 1:
                kernel = tinygp.transforms.Subspace(k, base_kernel)
            else:
                kernel = base_kernel
        else:
            raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return kernel

    def loglikelihood(self, fit_lc, channel=None):
        """Compute the GP log-likelihood.

        Parameters
        ----------
        fit_lc : ndarray
            The fitted (non-GP) model.
        channel : int; optional
            If provided, evaluate only that channel. Defaults to None.

        Returns
        -------
        float
            Log-likelihood from the selected GP backend.
        """
        nchan, channels = self._channels(channel)

        logL = 0.
        for chan in channels:
            if self.nchannel_fitted > 1:
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if channel is None:
                    fit_temp = split([fit_lc, ], self.nints, chan)[0]
                else:
                    # If only a specific channel is being evaluated, then only
                    # that channel's fitted model will be passed in
                    fit_temp = fit_lc
            else:
                # get flux and uncertainties for current channel
                flux = self.flux
                fit_temp = fit_lc
                unc_fit = self.unc_fit
            residuals = np.ma.masked_invalid(flux-fit_temp)
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time
            residuals = np.ma.masked_where(time.mask, residuals)

            # Remove poorly handled masked values
            good = ~np.ma.getmaskarray(residuals)

            # Subsampling for george and tinygp only
            do_subsample = self.gp_code_name in ('george', 'tinygp') and self.gp_subsample > 1
            if self.gp_code_name == 'tinygp':
                if self.kernel_inputs is None:
                    self.setup_inputs()
                kin_mask = np.ma.getmaskarray(
                    self.kernel_inputs[chan]).any(axis=0)
                good &= ~kin_mask
            # Choose the mask for subsampling or not
            if do_subsample:
                step = self.gp_subsample
                good_idx = np.flatnonzero(good)
                sub_idx = good_idx[::step]
                good_mask = np.zeros_like(good)
                good_mask[sub_idx] = True
            else:
                good_mask = good
            unc_good = unc_fit[good_mask]
            res_good = residuals[good_mask]
            gp = self.setup_GP(
                chan, good=good_mask,
                diag=np.asarray(unc_good)**2 if self.gp_code_name == 'tinygp' else None)

            if self.gp_code_name == 'george':
                # Select kernel inputs consistent with the chosen data mask
                kin = self.kernel_inputs[chan][:, good_mask].T
                if isinstance(kin, np.ma.MaskedArray):
                    kin = np.asarray(kin.data)
                else:
                    kin = np.asarray(kin)
                # Ensure uncertainties and residuals are plain ndarrays
                if isinstance(unc_good, np.ma.MaskedArray):
                    unc_good = np.asarray(unc_good.data)
                else:
                    unc_good = np.asarray(unc_good)
                res_arr = np.asarray(res_good)
                gp.compute(kin, unc_good)
                logL += gp.lnlikelihood(res_arr, quiet=True)
            elif self.gp_code_name == 'celerite':
                # celerite never subsamples (enforced in s5_meta),
                # but use good_mask for consistency
                kin = self.kernel_inputs[chan][0][good_mask]
                gp.compute(kin, yerr=unc_good)
                logL += gp.log_likelihood(res_good)
            elif self.gp_code_name == 'tinygp':
                if tinygp is None:
                    raise RuntimeError('tinygp is not available.')
                cond = gp.condition(np.asarray(res_good))
                logL += cond.log_probability
            else:
                raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return logL
