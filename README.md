# firas_distortions

Requirements: [sd_foregrounds](https://github.com/asabyr/sd_foregrounds), [numpyro](https://num.pyro.ai/en/latest/index.html#), [emcee](https://emcee.readthedocs.io/en/stable/), [getdist](https://getdist.readthedocs.io/en/latest/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/)

## Some notes about [/code](/code):

- emcee:
	- Inu.py, priors.py: models & priors for emcee
	- mcmc_funcs.py, run_mcmc.py: emcee sampling
- NUTS:
	- numpyro_models.py, numpyro_models_help.py, jax_sd_fg.py: models used in NUTS sampling.
	- numpyro_funcs.py, run_NUTs.py: NUTS sampling.
- minimization:
	- findbestfit.py: class to minimize (multiple starting points + iterative minimization)
	- minimize_priors.py: bounds for scipy minimizer
	- minimization_funcs_table.py: additional functions to compute chi^2 for various set-ups in the paper.
- plots:
	- contours.py: class to make all the contour plots and y-value horizontal plots.
- general:
	- read_data.py: functions to read FIRAS data
   	- read_nuts_posteriors.py: class to read .pkl posteriors from NUTS
	- help_funcs.py: mostly for labels
	- constants.py
 - SZ pack:
	- save_sz_pack_parallel.py, make_table_for_SZpack.py: running SZ pack and making a table to interpolate rel. effect
