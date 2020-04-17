import numpy as np

BASELINE = 0, 10


def baseline(pupil_column):
    """ Median pupil size during the first ten samples was taken as baseline pupil size.
        The given df should be the specific pupil diameter column. """
    # calculate the median of first examples
    return pupil_column.iloc[0:10].median()


def correction(df, base_line, subtraction=True):
    """ We will use 2 methods for correction of the pupils diameter:
            1. subtraction - subtract the baseline from each example.
            2. division - divide the whole column by the baseline.
        The given df should be the whole data df. """

    if subtraction is True:
        df['Diameter'] -= base_line
    else:
        df['Diameter'] /= base_line


def get_change(df):
    """ Percentage change between the current and a prior element. """
    df['Diameter'] = df['Diameter'].pct_change()


def hampel_filter_outliers(input_series, window_size=10, n_sigmas=3):
    """ The goal of the Hampel filter is to identify and replace outliers in a given series.
        Function for outlier detection using the Hampel filter.
        Based on `pracma` implementation in R.

        Parameters
        ------------
        input_series : np.ndarray
            The series on which outlier detection will be performed
        window_size : int
            The size of the window (one-side). Total window size is 2*window_size+1
        n_sigmas : int
            The number of standard deviations used for identifying outliers

        Returns
        -----------
        new_series : np.ndarray
            The array in which outliers were replaced with respective window medians
        indices : np.ndarray
            The array containing the indices of detected outliers
        """

    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    indices = []

    # possibly use np.nanmedian
    for i in range(window_size, (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if np.abs(input_series[i] - x0) > n_sigmas * S0:
            new_series[i] = x0
            indices.append(i)

    return new_series, indices
