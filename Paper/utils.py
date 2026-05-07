import numpy as np 
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt 

def fmt_pctl(x, order="50_16_84", ndp=1, latex=True):
	"""
	Format percentile tuple into 'median^{+hi}_{-lo}' (LaTeX) or 'median +hi -lo' (plain).

	Parameters
	----------
	x : tuple/list/np.ndarray or float
		Percentiles or scalar. If tuple-like: contains 16/50/84 in some order.
	order : {"50_16_84","16_50_84"}
		How x is ordered.
	ndp : int
		Decimal places.
	latex : bool
		If True -> LaTeX superscript/subscript format.
	"""
	x = np.asarray(x, dtype=float).ravel()
	if x.size != 3 or not np.all(np.isfinite(x)):
		return ""  # or raise ValueError

	if order == "50_16_84":
		p50, p16, p84 = x
	elif order == "16_50_84":
		p16, p50, p84 = x
	else:
		raise ValueError("order must be '50_16_84' or '16_50_84'")

	lo = p50 - p16
	hi = p84 - p50

	if latex:
		return rf"{p50:.{ndp}f}^{{+{hi:.{ndp}f}}}_{{-{lo:.{ndp}f}}}"
	else:
		return f"{p50:.{ndp}f} +{hi:.{ndp}f} -{lo:.{ndp}f}"



def median_p16_p84(x):
	x = np.asarray(x, float).ravel()
	x = x[np.isfinite(x)]
	if x.size == 0:
		return np.array([np.nan, np.nan, np.nan])
	return np.percentile(x, [50, 16, 84])
def alpha_from_N(Nmin,Nmax,N, a0=0.25, a1=0.85):
    if Nmax == Nmin:
        return a1
    return a0 + (a1 - a0) * (N - Nmin) / (Nmax - Nmin)

def _to_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    return x[m], y[m]

def rgba_by_density(x, y, base_color, alpha_min=0.30, alpha_max=1.00, power=0.7):
    """
    KDE density in (x, log10(y)) space -> alpha in [alpha_min, alpha_max].
    """
    x, y = _to_xy(x, y)
    if x.size < 10:
        rgba = np.tile(mcolors.to_rgba(base_color), (x.size, 1))
        rgba[:, 3] = alpha_max
        return x, y, rgba

    z = np.vstack([x, np.log10(y)])
    kde = gaussian_kde(z)
    dens = kde(z)

    # robust normalize density -> [0,1]
    lo, hi = np.percentile(dens, [5, 99])
    t = (dens - lo) / (hi - lo + 1e-12)
    t = np.clip(t, 0, 1)

    a = alpha_min + (alpha_max - alpha_min) * (t ** power)

    rgba = np.tile(mcolors.to_rgba(base_color), (x.size, 1))
    rgba[:, 3] = a
    return x, y, rgba

def legend_handle(color, marker, label, alpha=0.30, ms=8):
    # A representative low-density point for the legend (alpha ~ alpha_min)
    return plt.Line2D(
        [0], [0],
        marker=marker,
        linestyle="None",
        markerfacecolor=mcolors.to_rgba(color, alpha),
        markeredgecolor=mcolors.to_rgba(color, alpha),
        markersize=ms,
        label=label,
    )
def bins_centered_on_zero(x, nbins=60, clip=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.linspace(-1, 1, nbins + 1)

    if clip is None:
        m = max(abs(np.nanmin(x)), abs(np.nanmax(x)))
    else:
        m = float(abs(clip))
    if m == 0:
        m = 1.0

    w = (2 * m) / nbins
    edges = np.arange(-m - w/2, m + w, w)
    return edges