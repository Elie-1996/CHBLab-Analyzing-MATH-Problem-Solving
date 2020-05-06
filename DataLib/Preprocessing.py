from Data.GetData import Data

import matplotlib.pyplot as plt
import numpy as np

BASELINE = 0, 10


def pupil_preprocessing():
    """" Prepossessing of pupil Diameter """
    pupil_df = Data.pupil_data
    blinks_df = Data.blinks_data
    plt.plot(pupil_df['Diameter'], linewidth=1, markersize=3, label='raw')

    # Getting blinks above some confidence threshold
    blinks_df = blinks_df[blinks_df['confidence'] > 0.5]
    # print(blinks_df['confidence'])
    """ function for pupils diameter data"""
    # 1. remove blinks and interpolate the data
    # remove blinks (set as None)
    for index, row in blinks_df.iterrows():
        start_frame = row['start_frame_index']
        end_frame = row['end_frame_index']
        pupil_df['Diameter'].iloc[start_frame:end_frame] = None

    # interpolate None rows
    pupil_df['Diameter'] = pupil_df['Diameter'].interpolate(method='cubic')

    # 2. using hample filter for detection of outliers and interpolate them using median value of neighbors
    Diameter = pupil_df['Diameter'].to_numpy()
    new_series, detected_outliers = hampel_filter_outliers(Diameter, 10)
    pupil_df['Diameter'] = new_series

    # 3. gaussian filter on data
    smooth = pupil_df['Diameter'].rolling(window=5, win_type='gaussian', center=True).mean(std=0.5)
    pupil_df['Diameter'] = smooth

    # 4. base line correction
    base_value = baseline(pupil_df['Diameter'])
    correction(pupil_df, base_value, subtraction=True)

    print(pupil_df['Diameter'])
    plt.plot(pupil_df['Diameter'], linewidth=1, markersize=3, label='filtered')

    # 5. getting relative change in size
    get_change(pupil_df)
    plt.show()


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
