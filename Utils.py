import numpy as np
import os
from Visualization import AOI
from DataLib.GetData import Data

input_directory = os.path.join('Subjects', 'data')  # CSV files
background_images = [None, 'Question2.jpg', 'Question3.jpg', None]
WIDTHS =  [None, 2046, 1810, None]  # The width of image for each question (here 2046 width for Question 2)
HEIGHTS = [None, 1155, 1014, None]  # The height of image for each question (here 1014 width for Question 3)
subjects_dict = {
    # 'None' value means we skip the analysis of that question, and tuple (start time, end time) means we partake
    # these specific times (in seconds) to include in question
    # For example: given this data: '000_fixations': [None, (12, 43), (50, 90), None]
    # then we will skip the first question, include seconds 12 through 43 in the second
    # question, include seconds 50 through 90 in the third question and skip the fourth question
    # Note: If you would like to exclude a subject entirely - simply fill them with 'None' values, example to exclude subject 9:
    # '009_fixations': [None, None, None, None]
    '002_fixations':  [None, None, None, None],
    '003_fixations':  [None, (368, 480), (493, 613), None],
    '004_fixations':  [None, (251, 380), (394, 560), None],
    '005_fixations':  [None, (385, 523), (536, 649), None],
    '006_fixations':  [None, None, None, None],
    '007_fixations':  [None, (361, 450), (468, 620), None],
    '008_fixations':  [None, (560, 701), (720, 930), None],
    '009_fixations':  [None, (460, 550), (584, 714), None],
    '1000_fixations': [None, (203, 303), (316, 414), None]
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
