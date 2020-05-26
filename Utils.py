import numpy as np
from Visualization import AOI
from DataLib.GetData import Data


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
