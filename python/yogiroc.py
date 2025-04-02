import numpy as np
import pandas as pd
from scipy.stats import binom, beta
import matplotlib.pyplot as plt
from math import ceil, log10
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def prc_CI(successes, totals, p=(0.025, 0.975), res=0.001):
    """
    Compute the confidence interval for a call‐rate (precision or recall)
    using the binomial likelihood.

    Parameters
    ----------
    successes : array-like
        Number of correct calls (e.g. true positives).
    totals : array-like
        Total number of calls (e.g. true positives + false positives).
    p : tuple of floats, optional
        Confidence interval probability range (default is (0.025, 0.975)).
    res : float, optional
        Resolution at which to sample the call rates (default is 0.001).

    Returns
    -------
    ci : np.ndarray
        A two-dimensional array of shape (len(successes), len(p))
        where each row contains the estimated call rate (precision/recall)
        at the lower and upper confidence limits.
    """
    successes = np.atleast_1d(successes)
    totals = np.atleast_1d(totals)
    if len(successes) != len(totals):
        raise ValueError("Length of successes and totals must be equal.")
    ci_list = []
    # Create a grid of rates from 0 to 1
    rates = np.arange(0, 1 + res, res)
    for i, n in zip(successes, totals):
        # Compute the likelihood (binomial PMF) over all rates
        dens = binom.pmf(i, n, rates)
        # Compute the cumulative density (using Riemann sum approximation)
        cdf = np.concatenate(([0], np.cumsum(dens[1:] * res) / np.sum(dens * res)))
        # For each probability cutoff, find the maximum rate such that cdf < cutoff
        estimates = []
        for prob in p:
            idx = np.where(cdf < prob)[0]
            # If no index is found (should not happen since cdf[0]==0), use 0.
            if len(idx) == 0:
                estimates.append(0.0)
            else:
                estimates.append(rates[idx[-1]])
        ci_list.append(estimates)
    return np.array(ci_list)


def sample_rates_qd(i, n, N=1000, minQ=0, maxQ=1):
    """
    Quick and dirty sampling of rate parameters for a binomial distribution
    using rejection sampling.

    Parameters
    ----------
    i : int
        Numerator (number of successful events).
    n : int
        Denominator (total number of events).
    N : int, optional
        Desired number of samples to generate (default is 1000).
    minQ : float, optional
        Minimum constraint for the samples (default is 0).
    maxQ : float, optional
        Maximum constraint for the samples (default is 1).

    Returns
    -------
    samples : np.ndarray
        A vector of sampled rates.
    """
    max_density = binom.pmf(i, n, i/n)
    # First round: attempt N samples.
    rate_samples = np.random.uniform(minQ, maxQ, size=N)
    # For each candidate sample, generate a uniform random number between 0 and max_density.
    u = np.random.uniform(0, max_density, size=N)
    accepted = rate_samples[u < binom.pmf(i, n, rate_samples)]
    # If not enough samples, perform additional rejection sampling.
    if accepted.size < N:
        # Estimate additional sample size (2x the expected number needed).
        M = int(2 * ceil(N / (accepted.size / N)))
        rate_samples2 = np.random.uniform(minQ, maxQ, size=M)
        u2 = np.random.uniform(0, max_density, size=M)
        accepted2 = rate_samples2[u2 < binom.pmf(i, n, rate_samples2)]
        accepted = np.concatenate((accepted, accepted2))
    return accepted[:N]


def rej_sam(i, n, minQ=0, maxQ=1):
    """
    Rejection sampling method for a single rate with constraints.

    Parameters
    ----------
    i : int
        Numerator (number of successful events).
    n : int
        Denominator (total number of events).
    minQ : float, optional
        Minimum constraint for the sample (default is 0).
    maxQ : float, optional
        Maximum constraint for the sample (default is 1).

    Returns
    -------
    rate : float
        A sampled rate that meets the constraints.
    """
    x = np.random.uniform(minQ, maxQ)
    # Add shortcuts for extreme constraints.
    if minQ > i/n and binom.pmf(i, n, minQ) < 0.05:
        return minQ
    if maxQ < i/n and binom.pmf(i, n, maxQ) < 0.05:
        return maxQ
    # Rejection sampling loop.
    while np.random.uniform(0, binom.pmf(i, n, i/n)) > binom.pmf(i, n, x):
        x = np.random.uniform(minQ, maxQ)
    return x


def sample_rates(i, n, N=1000, minQ=None, maxQ=None):
    """
    Sampling of rate parameters for a binomial distribution.
    If no constraints are provided, samples from the Beta distribution.
    Otherwise uses rejection sampling with the provided constraints.

    Parameters
    ----------
    i : int
        Numerator (number of successful events).
    n : int
        Denominator (total number of events).
    N : int, optional
        Number of samples to generate (default is 1000).
    minQ : float or array-like or None, optional
        Minimum constraint for the samples (default is None).
    maxQ : float or array-like or None, optional
        Maximum constraint for the samples (default is None).

    Returns
    -------
    samples : np.ndarray
        A vector of sampled rates.
    """
    if minQ is None and maxQ is None:
        # Handle degenerate cases: if no successes or if all are successes.
        if i == 0:
            return np.zeros(N)
        elif i == n:
            return np.ones(N)
        else:
            return np.random.beta(a=i, b=n - i, size=N)
    else:
        # If constraints are provided, generate N samples using rejection sampling.
        # If minQ or maxQ are arrays, assume they are provided per sample.
        samples = np.empty(N)
        for idx in range(N):
            # For vectorized constraints, extract the current value.
            current_min = minQ[idx] if isinstance(minQ, (list, np.ndarray)) else (minQ if minQ is not None else 0)
            current_max = maxQ[idx] if isinstance(maxQ, (list, np.ndarray)) else (maxQ if maxQ is not None else 1)
            samples[idx] = rej_sam(i, n, minQ=current_min, maxQ=current_max)
        return samples


def monotonize(xs):
    """
    Enforce a non-decreasing (monotonic) order on the input vector.
    Each element that is lower than its predecessor is set equal to the previous value.

    Parameters
    ----------
    xs : array-like
        Input numerical vector.

    Returns
    -------
    mono_xs : np.ndarray
        Monotonized vector.
    """
    xs = np.array(xs, copy=True)
    for i in range(1, len(xs)):
        if xs[i] < xs[i - 1]:
            xs[i] = xs[i - 1]
    return xs


def balance_prec(ppv, prior):
    """
    Compute the balanced precision using the formula:
      balanced = ppv*(1-prior) / (ppv*(1-prior) + (1-ppv)*prior)

    Parameters
    ----------
    ppv : float or array-like
        Precision values.
    prior : float
        Prior probability.

    Returns
    -------
    balanced_ppv : float or np.ndarray
        Balanced precision values.
    """
    return ppv * (1 - prior) / (ppv * (1 - prior) + (1 - ppv) * prior)


def configure_prec(sheet, monotonized_flag=True, balanced_flag=False):
    """
    Helper function to adjust precision by applying optional prior balancing and/or monotonization.

    Parameters
    ----------
    sheet : pd.DataFrame
        Data table (e.g. a ROC/PRC table) with at least the columns 'ppv_prec', 'tp', and 'fp'.
    monotonized_flag : bool, optional
        Whether to monotonize the precision (default is True).
    balanced_flag : bool, optional
        Whether to use prior-balancing (default is False).

    Returns
    -------
    ppv : np.ndarray
        Configured precision values.
    """
    ppv = sheet["ppv_prec"].values
    if balanced_flag:
        # Use the first row of the table to determine the prior.
        prior = sheet.iloc[0]["tp"] / (sheet.iloc[0]["tp"] + sheet.iloc[0]["fp"])
        ppv = balance_prec(ppv, prior)
    if monotonized_flag:
        ppv = monotonize(ppv)
    return ppv


# -----------------------------------------------------------------------------
# Running function for binned summaries
# -----------------------------------------------------------------------------

def running_function(x, y, nbins=50, q=2.5):
    """
    Compute a running summary (e.g. percentile) of y values binned by x.

    Parameters
    ----------
    x : array-like
        x values (e.g. recall values).
    y : array-like
        y values (e.g. precision values).
    nbins : int, optional
        Number of bins to use (default is 50).
    q : float, optional
        Percentile to compute (default is 2.5 for the lower bound).

    Returns
    -------
    result : np.ndarray
        Array of shape (nbins, 2) where the first column is the bin center
        and the second column is the computed percentile of y within that bin.
    """
    x = np.array(x)
    y = np.array(y)
    bins = np.linspace(np.min(x), np.max(x), nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    result = []
    for i in range(nbins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(mask):
            value = np.percentile(y[mask], q)
        else:
            value = np.nan
        result.append([bin_centers[i], value])
    return np.array(result)


def infer_prc_CI(random_paths, nbins=50):
    """
    Use sampled PRC paths to infer confidence intervals along the recall axis.

    Parameters
    ----------
    random_paths : list of np.ndarray
        List of arrays (each of shape (N, 2)) containing sampled [precision, recall] paths.
    nbins : int, optional
        Number of bins along the recall axis (default is 50).

    Returns
    -------
    ci_df : pd.DataFrame
        A DataFrame with columns ["recall", "0.025", "0.975"] representing the
        recall values (bin centers) and the corresponding lower and upper precision bounds.
    """
    # Concatenate all sampled paths vertically.
    random_samples = np.vstack(random_paths)
    # Compute the lower (2.5th percentile) and upper (97.5th percentile) of precision
    # for bins along the recall axis.
    q5 = running_function(random_samples[:, 1], random_samples[:, 0], nbins=nbins, q=2.5)
    q95 = running_function(random_samples[:, 1], random_samples[:, 0], nbins=nbins, q=97.5)
    ci = np.column_stack((q5[:, 0], q5[:, 1], q95[:, 1]))
    ci_df = pd.DataFrame(ci, columns=["recall", "0.025", "0.975"])
    return ci_df


def sample_prcs(data, N=1000, monotonized=True, sr=sample_rates):
    """
    Sample a distribution of PRC paths based on likelihood dictated by data.

    Parameters
    ----------
    data : pd.DataFrame
        A data table (e.g., a ROC/PRC table) with columns "tp", "fp", and "fn".
    N : int, optional
        Desired number of samples to generate (default is 1000).
    monotonized : bool, optional
        Whether to enforce monotonization on the sampled paths (default is True).
    sr : function, optional
        Sampling function for rate parameters. Should have signature:
        sr(i, n, N=1000, minQ=None, maxQ=None). (Default is sample_rates.)

    Returns
    -------
    random_paths : list of np.ndarray
        A list where each element is an (N x 2) numpy array with two columns:
        the first column is sampled precision and the second is sampled recall.
    """
    random_paths = []
    # Process the first row separately.
    row0 = data.iloc[0]
    prec_samples = sr(int(row0['tp']), int(row0['tp']) + int(row0['fp']), N=N)
    recall_samples = sr(int(row0['tp']), int(row0['tp']) + int(row0['fn']), N=N)
    random_paths.append(np.column_stack((prec_samples, recall_samples)))
    
    # Iterate over rows 1 to len(data)-2 (similar to R's 2:(nrow(data)-1))
    for k in tqdm(range(1, len(data) - 1), desc="Sampling PRC paths"):
        rowk = data.iloc[k]
        if monotonized:
            prev_samples = random_paths[k - 1]
            # Use the previous samples as constraints.
            prec_samples = sr(int(rowk['tp']), int(rowk['tp']) + int(rowk['fp']),
                              N=N, minQ=prev_samples[:, 0])
            recall_samples = sr(int(rowk['tp']), int(rowk['tp']) + int(rowk['fn']),
                                N=N, maxQ=prev_samples[:, 1])
        else:
            prec_samples = sr(int(rowk['tp']), int(rowk['tp']) + int(rowk['fp']), N=N)
            recall_samples = sr(int(rowk['tp']), int(rowk['tp']) + int(rowk['fn']), N=N)
        random_paths.append(np.column_stack((prec_samples, recall_samples)))
    return random_paths


# -----------------------------------------------------------------------------
# YogiROC2 class (object constructor and methods)
# -----------------------------------------------------------------------------

class YogiROC2:
    """
    YogiROC2 object for computing ROC and PRC curves.

    Parameters
    ----------
    truth : array-like of bool
        Boolean vector indicating the true class for each observation.
    scores : pd.DataFrame or np.ndarray
        Matrix-like object where each row corresponds to an observation and each
        column to a predictor’s score.
    names : list of str, optional
        Names of the predictors. If not provided and scores is a DataFrame, its
        column names will be used.
    high : bool or list of bool, optional
        Indicates whether the scoring is high-to-low. If False (or for a given
        predictor, False), the corresponding scores are multiplied by -1.
        Default is True (i.e. higher scores mean more likely positive).

    Attributes
    ----------
    tables : dict
        Dictionary mapping predictor names to a DataFrame of ROC/PRC values.
        Each table has columns: 'thresh', 'tp', 'tn', 'fp', 'fn',
        'ppv_prec', 'tpr_sens', and 'fpr_fall'.
    """
    def __init__(self, truth, scores, names=None, high=True):
        # Ensure truth is a boolean array.
        truth = np.asarray(truth, dtype=bool)
        if isinstance(scores, np.ndarray):
            scores = pd.DataFrame(scores)
        elif not isinstance(scores, pd.DataFrame):
            raise ValueError("scores must be a pandas DataFrame or a numpy array.")
        n_obs, n_preds = scores.shape
        # Check dimensions.
        if len(truth) != n_obs:
            raise ValueError("Length of truth must equal the number of rows in scores.")
        if names is None:
            names = list(scores.columns)
        if len(names) != n_preds:
            raise ValueError("Length of names must equal the number of columns in scores.")
        # Process the 'high' argument.
        if isinstance(high, bool):
            if not high:
                scores = -scores
        else:
            # high is expected to be list-like (one bool per predictor).
            if len(high) != n_preds:
                raise ValueError("Length of high must be 1 or equal to number of predictors.")
            for idx, flag in enumerate(high):
                if not flag:
                    scores.iloc[:, idx] = -scores.iloc[:, idx]

        self.names = names
        self.truth = pd.Series(truth, index=scores.index)
        self.tables = {}
        # Compute ROC/PRC table for each predictor.
        for col in names:
            col_scores = scores[col]
            valid = col_scores.notna()
            truth_valid = self.truth[valid]
            # Compute sample prior as the fraction of true cases.
            prior = truth_valid.mean()
            # Use unique sorted thresholds including -inf and +inf.
            # thresholds = np.concatenate(([-np.inf], np.sort(col_scores.unique()), [np.inf]))
            # rows = []
            # for t in thresholds:
            #     # Determine which calls (scores) are above threshold.
            #     calls = col_scores >= t
            #     # Calculate confusion matrix components.
            #     tp = np.sum(calls & self.truth)
            #     tn = np.sum((~calls) & (~self.truth))
            #     fp = np.sum(calls & (~self.truth))
            #     fn = np.sum((~calls) & self.truth)
            #     # Calculate precision (PPV), sensitivity (TPR), and fallout (FPR).
            #     ppv_prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            #     tpr_sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            #     fpr_fall = fp / (tn + fp) if (tn + fp) > 0 else np.nan
            #     rows.append([t, tp, tn, fp, fn, ppv_prec, tpr_sens, fpr_fall])

            # Assume truth_valid and col_scores are your ground truth and scores arrays.
            precision, recall, pr_thresholds = precision_recall_curve(truth_valid, col_scores)
            df_prc = pd.DataFrame({
                "thresh_prc": np.concatenate(([-np.inf], pr_thresholds)),
                "ppv_prec": precision, 
                "tpr_sens": recall, 
                
            })
            # derive confusion matrix counts:
            fpr, tpr, roc_thresholds = roc_curve(truth_valid, col_scores)
            P = np.sum(truth_valid)                # total positives
            N = len(truth_valid) - P               # total negatives
            # Derive counts for each threshold from the ROC metrics:
            tp = tpr * P                         # True Positives
            fp = fpr * N                         # False Positives
            fn = P - tp                          # False Negatives
            tn = N - fp                          # True Negatives

            df_roc = pd.DataFrame(
                {
                    "thresh_roc": roc_thresholds,
                    "tp": tp, 
                    "tn": tn, 
                    "fp": fp, 
                    "fn": fn, 
                    "fpr_fall": fpr
                }
            )
            df = df_prc.merge(df_roc, left_on="thresh_prc", right_on="thresh_roc", how="outer")
            # Adjust the last row's precision to equal the penultimate row's value.
            # if len(df) >= 2:
            #     df.loc[df.index[-1], "ppv_prec"] = df.loc[df.index[-2], "ppv_prec"]
            self.tables[col] = df


    def __str__(self):
        # Use the first table to get the reference set size (excluding the -inf and +inf rows).
        first_table = next(iter(self.tables.values()))
        ref_size = len(first_table) - 2
        preds = ", ".join(self.names)
        return f"YogiROC object\nReference set size: {ref_size}\nPredictors: {preds}"


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def calc_auc(xs, ys):
    """
    Calculate the area under a curve using the trapezoidal rule.
    If the x-values are in descending order, they are reversed to ensure
    a positive area.
    
    Parameters
    ----------
    xs : array-like
        x values.
    ys : array-like
        Corresponding y values.
        
    Returns
    -------
    auc : float
        The computed area under the curve.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # If xs is descending, reverse both arrays.
    if xs[0] > xs[-1]:
        xs = xs[::-1]
        ys = ys[::-1]
    return np.trapz(ys, xs)


def auprc(yroc, monotonized_flag=True, balanced_flag=False):
    """
    Calculate the area under the Precision-Recall curve (AUPRC)
    for each predictor in a YogiROC2 object.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    monotonized_flag : bool, optional
        Whether to use a monotonized PRC (default is True).
    balanced_flag : bool, optional
        Whether to use prior-balancing (default is False).

    Returns
    -------
    auprc_dict : dict
        Dictionary mapping predictor names to their AUPRC values.
    """
    results = {}
    for name, table in yroc.tables.items():
        configured = configure_prec(table, monotonized_flag, balanced_flag)
        auc = calc_auc(table["tpr_sens"].values, configured)
        results[name] = auc
    return results


def auroc(yroc):
    """
    Calculate the area under the ROC curve (AUROC) for each predictor.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.

    Returns
    -------
    auroc_dict : dict
        Dictionary mapping predictor names to their AUROC values.
    """
    results = {}
    for name, table in yroc.tables.items():
        auc = calc_auc(table["fpr_fall"].values, table["tpr_sens"].values)
        results[name] = auc
    return results


def recall_at_prec(yroc, x=0.9, monotonized_flag=True, balanced_flag=False):
    """
    Calculate the maximum recall (TPR) at a given minimum precision level.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    x : float, optional
        The precision cutoff (default is 0.9).
    monotonized_flag : bool, optional
        Whether to use a monotonized PRC (default is True).
    balanced_flag : bool, optional
        Whether to use prior-balancing (default is False).

    Returns
    -------
    recall_dict : dict
        Dictionary mapping predictor names to the maximum recall achieving at least
        precision x (or np.nan if none).
    """
    results = {}
    for name, table in yroc.tables.items():
        ppv = configure_prec(table, monotonized_flag, balanced_flag)
        valid = ppv > x
        if np.any(valid):
            results[name] = np.max(table["tpr_sens"].values[valid])
        else:
            results[name] = np.nan
    return results


def draw_roc(yroc, col=None, lty=None, legend_pos='lower right', **kwargs):
    """
    Draw ROC curves for a YogiROC2 object.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    col : list, optional
        List of colors for each predictor. If None, defaults are used.
    lty : list, optional
        List of line style strings for each predictor (e.g., '-', '--').
    legend_pos : str, optional
        Position of the legend (default is 'lower right'); pass None to disable legend.
    **kwargs :
        Additional keyword arguments to pass to matplotlib's plot function.

    Returns
    -------
    None
    """
    plt.figure()
    pred_names = yroc.names
    n = len(pred_names)
    if col is None:
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    if lty is None:
        lty = ['-'] * n
    # Plot the first predictor.
    table = yroc.tables[pred_names[0]]
    plt.plot(100 * table["fpr_fall"], 100 * table["tpr_sens"],
             color=col[0], linestyle=lty[0], label=pred_names[0], **kwargs)
    # Plot subsequent predictors.
    for i in range(1, n):
        table = yroc.tables[pred_names[i]]
        plt.plot(100 * table["fpr_fall"], 100 * table["tpr_sens"],
                 color=col[i], linestyle=lty[i], label=pred_names[i], **kwargs)
    # Add legend with AUROC values.
    auroc_vals = auroc(yroc)
    if legend_pos is not None:
        leg_labels = [f"{name} (AUROC={auroc_vals[name]:.2f})" for name in pred_names]
        plt.legend(leg_labels, loc=legend_pos)
    plt.xlabel("False positive rate (%)\n(= 100% - specificity)")
    plt.ylabel("Sensitivity or True positive rate (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("ROC Curves")
    plt.show()


def draw_prc(yroc, col=None, lty=None, monotonized_flag=True,
             balanced_flag=False, legend_pos='lower left', **kwargs):
    """
    Draw Precision-Recall (PRC) curves for a YogiROC2 object.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    col : list, optional
        List of colors for each predictor (default uses matplotlib defaults).
    lty : list, optional
        List of line style strings for each predictor (default is solid lines).
    monotonized_flag : bool, optional
        Whether to monotonize the precision (default is True).
    balanced_flag : bool, optional
        Whether to use prior-balancing (default is False).
    legend_pos : str, optional
        Position of the legend (default is 'lower left'); pass None to disable legend.
    **kwargs :
        Additional keyword arguments for the plot.

    Returns
    -------
    None
    """
    plt.figure()
    pred_names = yroc.names
    n = len(pred_names)
    if col is None:
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    if lty is None:
        lty = ['-'] * n

    # Helper lambda to get configured precision.
    get_ppv = lambda name: configure_prec(yroc.tables[name], monotonized_flag, balanced_flag)

    # Plot the first predictor.
    table = yroc.tables[pred_names[0]]
    plt.plot(100 * table["tpr_sens"], 100 * get_ppv(pred_names[0]),
             color=col[0], linestyle=lty[0], label=pred_names[0], **kwargs)
    # Plot subsequent predictors.
    for i in range(1, n):
        table = yroc.tables[pred_names[i]]
        plt.plot(100 * table["tpr_sens"], 100 * get_ppv(pred_names[i]),
                 color=col[i], linestyle=lty[i], label=pred_names[i], **kwargs)
    # Add legend with AUPRC and R90P.
    auprc_vals = auprc(yroc, monotonized_flag, balanced_flag)
    r90p_vals = recall_at_prec(yroc, x=0.9, monotonized_flag=monotonized_flag, balanced_flag=balanced_flag)
    if legend_pos is not None:
        leg_labels = [f"{name} (AUPRC={auprc_vals[name]:.2f}; R90P={r90p_vals[name]:.2f})"
                      for name in pred_names]
        plt.legend(leg_labels, loc=legend_pos)
    ylabel = "Balanced precision (%)" if balanced_flag else "Precision (%)"
    plt.xlabel("Recall (%)")
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Precision-Recall Curves")
    plt.show()


def draw_prc_CI(yroc, col=None, lty=None, monotonized_flag=True,
                balanced_flag=False, legend_pos='lower left',
                sampling="accurate", nsamples=1000, monotonizedSampling=False, **kwargs):
    """
    Draw a Precision-Recall Curve with confidence intervals using sampling.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    col : list, optional
        List of colors for each predictor (default uses matplotlib defaults).
    lty : list, optional
        List of line style strings for each predictor (default is solid lines).
    monotonized_flag : bool, optional
        Whether to use monotonized precision for the curve (default is True).
    balanced_flag : bool, optional
        Whether to use prior-balancing (default is False).
    legend_pos : str, optional
        Legend position (default is 'lower left'); set to None to disable legend.
    sampling : str, optional
        Sampling method: "accurate" uses sample_rates, "quickDirty" uses sample_rates_qd.
    nsamples : int, optional
        Number of samples to use in the PRC confidence interval (default is 1000).
    monotonizedSampling : bool, optional
        Whether to monotonize during sampling (default is False).
    **kwargs :
        Additional keyword arguments for the plot.

    Returns
    -------
    None
    """
    plt.figure()
    pred_names = yroc.names
    n = len(pred_names)
    if col is None:
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n]
    if lty is None:
        lty = ['-'] * n
    # Choose sampling function.
    if sampling == "quickDirty":
        sr = sample_rates_qd
    else:
        sr = sample_rates

    get_ppv = lambda name: configure_prec(yroc.tables[name], monotonized_flag, balanced_flag)

    # Plot the PRC curves.
    table0 = yroc.tables[pred_names[0]]
    plt.plot(100 * table0["tpr_sens"], 100 * get_ppv(pred_names[0]),
             color=col[0], linestyle=lty[0], label=pred_names[0], **kwargs)
    for i in range(1, n):
        table_i = yroc.tables[pred_names[i]]
        plt.plot(100 * table_i["tpr_sens"], 100 * get_ppv(pred_names[i]),
                 color=col[i], linestyle=lty[i], label=pred_names[i], **kwargs)
    # For each predictor, sample PRC paths and infer confidence intervals.
    for name, clr in zip(pred_names, col):
        # Use the first row of the table to compute the prior.
        table = yroc.tables[name]
        prior = table.iloc[0]["tp"] / (table.iloc[0]["tp"] + table.iloc[0]["fp"])
        # Sample paths.
        sampled_paths = sample_prcs(table, N=nsamples, monotonized=monotonizedSampling, sr=sr)
        ci_df = infer_prc_CI(sampled_paths, nbins=50)
        # Optionally apply balancing to the CI columns.
        for col_ci in ["0.025", "0.975"]:
            ci_df[col_ci] = np.where(np.isnan(ci_df[col_ci]),
                                       ci_df[col_ci],
                                       balance_prec(ci_df[col_ci], prior) if balanced_flag else ci_df[col_ci])
        # Fill the area between the lower and upper CI.
        plt.fill_between(100 * ci_df["recall"],
                         100 * ci_df["0.025"],
                         100 * ci_df["0.975"],
                         color=clr, alpha=0.1)
    auprc_vals = auprc(yroc, monotonized_flag, balanced_flag)
    r90p_vals = recall_at_prec(yroc, x=0.9, monotonized_flag=monotonized_flag, balanced_flag=balanced_flag)
    if legend_pos is not None:
        leg_labels = [f"{name} (AUPRC={auprc_vals[name]:.2f}; R90P={r90p_vals[name]:.2f})"
                      for name in pred_names]
        plt.legend(leg_labels, loc=legend_pos)
    ylabel = "Balanced precision (%)" if balanced_flag else "Precision (%)"
    plt.xlabel("Recall (%)")
    plt.ylabel(ylabel)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Precision-Recall Curve with Confidence Intervals")
    plt.show()


# -----------------------------------------------------------------------------
# Significance functions for AUPRC
# -----------------------------------------------------------------------------

def auprc_signif(yroc, monotonized_flag=True, res=0.001):
    """
    Assess the significance of AUPRC differences among predictors.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    monotonized_flag : bool, optional
        Whether to use a monotonized PRC curve (default is True).
    res : float, optional
        The resolution at which to sample the probability function (default is 0.001).

    Returns
    -------
    result : dict
        A dictionary with the following keys:
            - 'auprc': empirical AUPRC for each predictor.
            - 'ci': 95% confidence intervals (2.5th and 97.5th percentiles) for AUPRC.
            - 'llr': log likelihood ratio matrix.
            - 'pval': p-value matrix for each predictor comparison.
    """
    ps = np.arange(res, 1 - res, res)
    pred_names = yroc.names
    n_preds = len(pred_names)
    # Compute a distribution of AUPRC values for each predictor.
    # For each predictor, compute a vector of AUPRC values over the range of ps.
    auprcs_mat = []
    for name in pred_names:
        table = yroc.tables[name]
        # Use prc_CI on each row (using tp and tp+fp)
        precCI = prc_CI(table["tp"].values, (table["tp"] + table["fp"]).values, p=tuple(ps), res=res)
        # For each probability level (each column), compute AUPRC using the corresponding precision vector.
        aucs = []
        for j in range(precCI.shape[1]):
            ppv = precCI[:, j]
            if monotonized_flag:
                ppv = monotonize(ppv)
            aucs.append(calc_auc(table["tpr_sens"].values, ppv))
        auprcs_mat.append(aucs)
    auprcs_mat = np.array(auprcs_mat).T  # shape: (len(ps), n_preds)
    # Empirical AUPRC values.
    empAUCs = np.array(list(auprc(yroc, monotonized_flag).values()))
    # Build a reverse-lookup table.
    auc_min = round(np.nanmin(auprcs_mat), 2)
    auc_max = round(np.nanmax(auprcs_mat), 2)
    aucRange = np.arange(auc_min, auc_max + res, res)
    # For each predictor, for each value in aucRange, find the corresponding p-value.
    aucPs = np.zeros((len(aucRange), n_preds))
    lookup = np.concatenate(([0], ps))
    for j in range(n_preds):
        for idx, a in enumerate(aucRange):
            count = np.sum(auprcs_mat[:, j] < a)
            # Use count as index into lookup.
            aucPs[idx, j] = lookup[min(count, len(lookup)-1)]
    # Confidence intervals for each predictor.
    confInts = {}
    for j, name in enumerate(pred_names):
        confInts[name] = np.percentile(auprcs_mat[:, j], [2.5, 97.5])
    # p-value matrix.
    pval = np.empty((n_preds, n_preds))
    for i in range(n_preds):
        for j in range(n_preds):
            if i == j:
                pval[i, j] = np.nan
            else:
                count = np.sum(auprcs_mat[:, j] < empAUCs[i])
                pval[i, j] = 1 - lookup[min(count, len(lookup)-1)]
    # Log likelihood ratios.
    llr = np.empty((n_preds, n_preds))
    for i in range(n_preds):
        for j in range(n_preds):
            if i == j:
                llr[i, j] = np.nan
            else:
                num = calc_auc(aucRange, aucPs[:, j] * (1 - aucPs[:, i]))
                den = calc_auc(aucRange, aucPs[:, i] * (1 - aucPs[:, j]))
                llr[i, j] = log10(num / den) if den > 0 else np.nan
    return {"auprc": empAUCs,
            "ci": confInts,
            "llr": llr,
            "pval": pval}


def auprc_pvrandom(yroc, monotonized_flag=True, cycles=10000):
    """
    Assess the significance of AUPRC against random guessing.

    Parameters
    ----------
    yroc : YogiROC2
        A YogiROC2 object.
    monotonized_flag : bool, optional
        Whether to use monotonized PRC curves (default is True).
    cycles : int, optional
        Number of cycles (iterations) to use for generating the null distribution (default is 10000).

    Returns
    -------
    pvals : dict
        Dictionary mapping predictor names to the empirical p-value against random guessing.
    """
    # Get empirical AUPRC values.
    empAUCs = np.array(list(auprc(yroc, monotonized_flag).values()))
    # Reconstruct the truth table from the first predictor.
    table0 = next(iter(yroc.tables.values()))
    real = int(table0.iloc[0]["tp"])
    nreal = int(table0.iloc[0]["fp"])
    truth = np.array([True] * real + [False] * nreal)
    nullAucs = []
    for _ in tqdm(range(cycles), desc="Sampling null AUPRC"):
        scores = np.random.uniform(0, 1, size=real + nreal)
        # Create thresholds.
        thresholds = np.concatenate(([-np.inf], np.sort(scores), [np.inf]))
        pr = []
        for t in np.sort(scores):
            calls = scores >= t
            tp = np.sum(calls & truth)
            fp = np.sum(calls & (~truth))
            fn = np.sum((~calls) & truth)
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            pr.append([prec, recall])
        pr = np.array(pr)
        if monotonized_flag and pr.size > 0:
            pr[:, 0] = monotonize(pr[:, 0])
        nullAucs.append(calc_auc(pr[:, 1], pr[:, 0]))
    nullAucs = np.array(nullAucs)
    pvals = {}
    for name, eauc in zip(yroc.names, empAUCs):
        pvals[name] = np.mean(nullAucs >= eauc)
    return pvals