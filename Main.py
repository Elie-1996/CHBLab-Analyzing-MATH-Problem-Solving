import Analysis
from Visualization import Visualization, VisualizationMap
from Data import Data, load_input_data

# TODO The x,y we get from norm_pos are normalize, we should get the real coordinate by multiplying each of them by
#  the length and width of the screen size
""" for eli - "df" is our main data frame, "pupil_data" is a data frame that includes additional info about the pupil
            df["RightX"] is the x coordinates of the right eye
            df["LeftX"] is the y coordinates of the right eye
            df["Diameter"] is the size of the right pupil
            df["Timestampe"] is the current frame Time stamp"""

if __name__ == '__main__':
    load_input_data()
    df = Data.normalized_df

    # converting to numpy array for later use
    time_array = df['Timestamp'].to_numpy()
    x_array = df['RightX'].to_numpy()
    y_array = df['RightY'].to_numpy()

    # TODO: the if and else here will be removed once we are finished with the testing phase of HeatMap Implementation
    #  pushed with testing_visualization = False to keep the old behaviour of the program on Master.
    testing_visualization = False
    if not testing_visualization:
        # some visualization (heat map)
        Visualization.scatter_density(df)
        # finding fixations - more info about Sfix Efix in Analysis module
        Sfix, Efix = Analysis.fixation_detection(x_array, y_array, time_array)
        # find saccades - more info about Ssac Esac in Analysis module
        Ssac, Esac = Analysis.saccade_detection(x_array, y_array, time_array)

        print(df)
    else:
        import numpy as np
        image = np.round(np.random.rand(30, 30) * 255) % 2
        image_contrast_stretched = np.round(255 * ((image - 0) / (2 - 0)))

        should_display_image = False
        if should_display_image:
            import matplotlib.pyplot as plt

            plt.imshow(image)
            plt.show()

        VisualizationMap(image_contrast_stretched, df, 2, 3)
