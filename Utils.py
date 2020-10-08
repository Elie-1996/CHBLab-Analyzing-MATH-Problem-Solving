import numpy as np
import os
from Visualization import AOI
from DataLib.GetData import Data

input_fixations_directory = os.path.join('Subjects', 'fixations')  # CSV files
input_blinks_directory = os.path.join('Subjects', 'blinks')  # CSV files
input_pupil_directory = os.path.join('Subjects', 'pupil')  # CSV files
background_images = [None, 'Question2.jpg', 'Question3.jpg', None]
WIDTHS = [None, 2046, 1810, None]  # The width of image for each question (here 2046 width for Question 2)
HEIGHTS = [None, 1155, 1014, None]  # The height of image for each question (here 1014 width for Question 3)
subjects_dict = {
    # 'None' value means we skip the analysis of that question, and tuple (start time, end time) means we partake
    # these specific times (in seconds) to include in question
    # For example: given this data: '000': [None, (12, 43), (50, 90), None]
    # then we will skip the first question, include seconds 12 through 43 in the second
    # question, include seconds 50 through 90 in the third question and skip the fourth question
    # Note: If you would like to exclude a subject entirely -
    # simply fill them with 'None' values, example to exclude subject 9:
    # '009': [None, None, None, None]
    # ##########################################################33
    '001': [None, None, None, None],
    '002':  [None, None, None, None],
    '003':  [None, None, None, None],
    '004':  [None, None, None, None],
    '005':  [None, (418, 435), None, None],
    '006':  [None, None, None, None],
    '007':  [None, None, None, None],
    '008':  [None, None, None, None],
    '009':  [None, None, None, None],
    '1000': [None, None, None, None],
}


def get_random_array_with_range(shape, min_range, max_range):
    return np.random.rand(shape) * (max_range - min_range) + min_range


def match_fixation_to_aoi():
    """
    iterate over fixations df, check if the fixations match one of the AOI's and assign if needed
    """
    fixations_df = Data.read_only_fixation_data(get_normalized=False)
    fixations_df["AOI"] = None

    # Insert fixations to the matching AOI
    for i, row in fixations_df.iterrows():
        for aoi_num, bound in enumerate(AOI.AOI_dict["1"]):
            if bound[0][0] <= row['X'] <= bound[3][0] and bound[3][1] <= row['Y'] <= bound[0][1]:
                # print("AOI Detected")
                fixations_df['AOI'].iloc[i] = aoi_num
                break

    # for i, row in fixations_df.iterrows():
    #     if row['AOI'] is not None:
    #         print(row['AOI'])
