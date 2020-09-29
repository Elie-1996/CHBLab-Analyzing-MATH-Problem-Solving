from DataLib import GetData

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from Utils import input_blinks_directory, input_pupil_directory, subjects_dict


df_interpolation_possibilities = ['cubic', 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'barycentric', 'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']

############################################################################
BASELINE = 0, 10
SUBJECT = "003"  # which subject to preprocess
QUESTION_IDX = 2  # which question to take (this is index and not value! which means IDX 0 is question 1)
NEIGHBOR_WINDOW_SIZE = 3  # This is a 3x3 window
HAMPEL_SIGMA = 3
HAMPEL_WINDOW_SIZE = 12
DF_INTERPOLATION = df_interpolation_possibilities[0]


def pupil_preprocessing():
    """" Prepossessing of pupil Diameter """
    pupil_df = pd.read_csv(os.path.join('..', input_pupil_directory, SUBJECT + '_pupil_positions.csv'))
    blinks_df = pd.read_csv(os.path.join('..', input_blinks_directory, SUBJECT + '_blinks.csv'))
    start_time, end_time = subjects_dict[SUBJECT + "_fixations"][QUESTION_IDX]

    pupil_df = pupil_df.rename(columns={'diameter': 'Diameter'}, inplace=False)
    pupil_df.pupil_timestamp -= pupil_df.pupil_timestamp.iloc[0]
    pupil_df = pupil_df.drop(pupil_df[(pupil_df.pupil_timestamp < start_time) | (pupil_df.pupil_timestamp > end_time)].index)
    plt.plot(pupil_df['Diameter'], linewidth=1, markersize=3, label='raw')

    # Getting blinks above some confidence threshold
    blinks_df = blinks_df[blinks_df['confidence'] > 0.5]
    # print(blinks_df['confidence'])
    """ function for pupils diameter data"""
    # 1. remove blinks and interpolate the data
    # remove blinks (set as None)
    old = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None
    for index, row in blinks_df.iterrows():
        start_frame = row['start_frame_index']
        end_frame = row['end_frame_index']
        pupil_df['Diameter'][start_frame:end_frame] = None
    pd.options.mode.chained_assignment = old

    # interpolate None rows
    pupil_df['Diameter'] = pupil_df['Diameter'].interpolate(method=DF_INTERPOLATION)

    # 2. using hample filter for detection of outliers and interpolate them using median value of neighbors
    Diameter = pupil_df['Diameter'].to_numpy()
    new_series, detected_outliers = hampel_filter_outliers(Diameter, HAMPEL_WINDOW_SIZE, HAMPEL_SIGMA)
    pupil_df['Diameter'] = new_series

    # 3. gaussian filter on data
    pupil_df['Diameter'] = pupil_df['Diameter'].iloc[3:].rolling(window=NEIGHBOR_WINDOW_SIZE, win_type='gaussian').mean(std=100)
    smooth_mean = pupil_df['Diameter'].mean()
    pupil_df = pupil_df.fillna(value={'Diameter': smooth_mean})

    # Cleaning really noisy places
    epsilon = 1
    for i, row in pupil_df.iterrows():
        # if the value is far from the mean by more then epsilon we will replace it with the previous value
        if not abs(smooth_mean - row['Diameter']) <= epsilon:
            row['Diameter'] = smooth_mean

    # 4. base line correction
    base_value = baseline(pupil_df['Diameter'])
    correction(pupil_df, base_value, subtraction=True)

    plt.plot(pupil_df['Diameter'], linewidth=1, markersize=3, label='filtered')

    # 5. get relative change in size (PCT)
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

############################################################################


pupil_preprocessing()
