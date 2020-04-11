import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from math import floor, ceil
import numpy


# TODO: Should create GUI using Tkinter to visualize the graph later on (Timestamp can control where the user looked
#  at that specific timestamp, scrollbar to see different timestamp, visualize other data, etc.)
# TODO: Information on Tkinter: https://www.geeksforgeeks.org/python-gui-tkinter/
# TODO: Integrating Tkinter and Matplotlib: https://www.youtube.com/watch?v=JQ7QP5rPvjU
class VisualizationMap:

    def __init__(self, image, df_as_dictionary,
                 horizontal_bins=5, vertical_bins=5,
                 initial_interval_start=0, initial_interval_end=5,
                 pad_value=float("-inf")):
        self.x_coords = df_as_dictionary['RightX']
        self.y_coords = df_as_dictionary['RightY']
        self.time_stamps = df_as_dictionary['Timestamp']
        self.full_image = np.array(image)
        self.horizontal_bins = horizontal_bins
        self.vertical_bins = vertical_bins
        self.bins = [[] for i in range(horizontal_bins * vertical_bins)]
        self.image_parts = self.__split_image_to_bins(
            self.full_image,
            horizontal_bins,
            vertical_bins,
            pad_value
        )
        self.time_interval_start = 0
        self.time_interval_end = 0
        self.update_interval(initial_interval_start, initial_interval_end)

    def update_interval(self, time_interval_start, time_interval_end):
        self.time_interval_start = time_interval_start
        self.time_interval_end = time_interval_end
        self.update_bin_division(self.horizontal_bins, self.vertical_bins)

    def update_bin_division(self, horizontal_bins, vertical_bins):
        self.horizontal_bins, self.vertical_bins = horizontal_bins, vertical_bins

        # x_coords and y_coords denote the coordinates that were allocated in the interval time stamp given by self.
        x_coords = self.__filter_data_according_to_time_stamp(self.x_coords)
        y_coords = self.__filter_data_according_to_time_stamp(self.y_coords)

        # first, clear the bins
        for _bin in self.bins:
            _bin.clear()
        self.bins.clear()
        self.bins = [[] for i in range(self.horizontal_bins * self.vertical_bins)]

        # second, update the bins
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            bin_idx = self.map_coordinate_to_bin_idx(x, y)
            self.bins[bin_idx].append((x, y))

    def __filter_data_according_to_time_stamp(self, coordinates):
        time_stamp = self.time_stamps
        coordinates_within_timestamp = []
        for i in range(len(time_stamp)):
            if self.time_interval_start <= time_stamp[i] <= self.time_interval_end:
                coordinates_within_timestamp.append(coordinates[i])
        return coordinates_within_timestamp

    # returns the index of the bin that (x, y) is within in self.bins
    # note: must be in sync with @__split_image_to_bins
    def map_coordinate_to_bin_idx(self, x, y):
        rows_amount, cols_amount = self.full_image.shape

        horizontal_jump = ceil(cols_amount / self.horizontal_bins)
        amount_of_bins_to_the_right = floor(x / horizontal_jump)

        vertical_jump = ceil(rows_amount / self.vertical_bins)
        amount_of_bins_to_down = floor(y / vertical_jump)

        return self.horizontal_bins * amount_of_bins_to_down + amount_of_bins_to_the_right

    # split any numpy image to sub-images, in a uniform manner according to horizontal_split and vertical_split.
    # horizontal_split, vertical_split - are integers in which specify how many splits to do in their direction.
    # design decision: if image pixels do not divide properly, the image will be enlarged to divide properly,
    # padded with pad_value.
    # Example: ([0, 1, 2] -> can't be divided into 2 equivalent parts -> [0, 1, 2, pad_value] -> can be divided into
    # 2 equivalent parts -> [0, 1], [2, pad_value])
    # note: must be in sync with @map_coordinate_to_bin_idx
    @staticmethod
    def __split_image_to_bins(image, horizontal_bins, vertical_bins, pad_value):
        original_image_rows, original_image_columns = image.shape
        image_rows, image_columns = image.shape

        # make sure that all blocks can fit an image.
        image, image_rows, image_columns = VisualizationMap.__pad_array(
            image,
            (horizontal_bins - (image_columns % horizontal_bins)) % horizontal_bins,
            (vertical_bins - (image_rows % vertical_bins)) % vertical_bins,
            pad_value
        )

        blocks_vertical_amount = int(image_rows / vertical_bins)
        blocks_horizontal_amount = int(image_columns / horizontal_bins)

        parts = list()
        for i in range(0, vertical_bins):
            for j in range(0, horizontal_bins):
                start_i, end_i = i * blocks_vertical_amount, (i + 1) * blocks_vertical_amount
                start_j, end_j = j * blocks_horizontal_amount, (j + 1) * blocks_horizontal_amount
                if start_i >= original_image_rows or start_j >= original_image_columns:
                    continue
                parts.append(image[start_i:end_i, start_j:end_j])

        return parts

    @staticmethod
    def __pad_array(image, horizontal_padding, vertical_padding, pad_value):
        image_rows, image_columns = image.shape
        # horizontal padding
        image = np.concatenate((image, pad_value * np.ones((image_rows, horizontal_padding))), axis=1)
        image_rows, image_columns = image.shape
        # vertical padding
        image = np.concatenate((image, pad_value * np.ones((vertical_padding, image_columns))), axis=0)
        image_rows, image_columns = image.shape

        return image, image_rows, image_columns

    # For debugging purposes
    def display_bin_data(self):
        for i in range(len(self.bins)):
            print("bin #" + str(i) + ": " + str(self.bins[i]))


class Visualization:

    @staticmethod
    def scatter_density(df):
        x = df['RightX']
        y = df['RightY']

        # ################ Plotting ####################################################
        g6 = plt.figure(1)
        ax6 = g6.add_subplot(111)
        xy = numpy.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.hist2d(x, y, (40, 40), cmap=plt.jet())
        plt.colorbar()
        plt.tick_params(labelsize=10)
        plt.title("Data density plot")
        plt.xlabel('Gaze coordinates (X) in pixels', fontsize=12)
        plt.ylabel('Gaze coordinates (Y) in pixels', fontsize=12)
        plt.tick_params(labelsize=16)
        plt.show()
