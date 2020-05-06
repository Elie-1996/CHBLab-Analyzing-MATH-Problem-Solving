import Analysis
from Utils import get_random_array_with_range
from Data import Data, load_input_data
from Visualization import VisualizationMap

if __name__ == '__main__':
    load_input_data(
        pupildata_dir='./000/pupil.pldata',
        gazedata_dir='./000/gaze.pldata',
        surface_fixation_dir='./000/exports/000/surfaces/fixations_on_surface_Surface 1.csv',
        blinks_data_dir='./000/exports/000/blinks.csv'
    )

    # TODO: Need to integrate the cleaning function within the "Data" class.
    if False:
        # clean and analyze pupil diameter value
        Analysis.pupils_preprocess(Data.pupil_data, Data.blinks_data)

    # TODO: the if and else here will be removed once we are finished with the testing phase of HeatMap Implementation
    #  pushed with testing_visualization = False to keep the old behaviour of the program on Master.
    testing_visualization = True
    if not testing_visualization:
        real_x = Data.gaze_data['X']
        real_y = Data.gaze_data['Y']
        time_array = Data.gaze_data['Timestamp']
        Ssac, Esac = Analysis.saccade_detection(real_x, real_y, time_array)

        fixation_list = list(map(lambda x_coord, y_coord: [x_coord, y_coord], Data.fixation_data['X'], Data.fixation_data['Y']))
        clusters = Analysis.making_clusters(fixation_list)

    else:
        import numpy as np
        import GUI

        image = np.round(np.random.rand(30, 30) * 255) % 2
        image_contrast_stretched = np.round(255 * ((image - 0) / (2 - 0)))

        should_display_image = False
        if should_display_image:
            import matplotlib.pyplot as plt

            plt.imshow(image)
            plt.show()

        test_df = {"X": [], "Y": [], 'Timestamp': np.arange(0, 20, 0.5)}  # 20/0.5 = 40 Samples!
        # x coordinates
        a = get_random_array_with_range(10, 0, 14)
        for x in a: test_df['X'].append(x)
        a = get_random_array_with_range(10, 16, 28)
        for x in a: test_df['X'].append(x)
        a = get_random_array_with_range(10, 0, 14)
        for x in a: test_df['X'].append(x)
        a = get_random_array_with_range(10, 16, 28)
        for x in a: test_df['X'].append(x)

        # y coordinates
        a = get_random_array_with_range(10, 0, 9)
        for x in a: test_df['Y'].append(x)
        a = get_random_array_with_range(10, 11, 18)
        for x in a: test_df['Y'].append(x)
        a = get_random_array_with_range(10, 21, 28)
        for x in a: test_df['Y'].append(x)
        a = get_random_array_with_range(10, 0, 9)
        for x in a: test_df['Y'].append(x)
        vm = VisualizationMap(image_contrast_stretched, test_df, 2, 3, 0, 20)
        vm.display_bin_data()
        GUI.setup_gui(vm)
