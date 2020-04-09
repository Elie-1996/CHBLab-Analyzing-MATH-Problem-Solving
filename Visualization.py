import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import numpy


# TODO: Should create GUI using Tkinter to visualize the graph later on (Timestamp can control where the user looked
#  at that specific timestamp, scrollbar to see different timestamp, visualize other data, etc.)
# TODO: Information on Tkinter: https://www.geeksforgeeks.org/python-gui-tkinter/
# TODO: Integrating Tkinter and Matplotlib: https://www.youtube.com/watch?v=JQ7QP5rPvjU
class VisualizationMap:
    """
        This class is expected to be a super class of other Visualization maps (or anything similar) in the future.
        It can be even further abstracted, for example: instead of rectangular bins, use hexagon bins. etc.
        So far, we are expecting HeatMaps and GraphMaps to be subclasses.
    """

    def __init__(self, image, df, horizontal_bins=5, vertical_bins=5, pad_value=float("-inf")):
        self.df = df
        self.full_image = np.array(image)
        self.image_parts = self.__split_image_to_rectangular_bins(
            self.full_image,
            horizontal_bins,
            vertical_bins,
            pad_value
        )

    # split any numpy image to sub-images, in a uniform manner according to horizontal_split and vertical_split.
    # horizontal_split, vertical_split - are integers in which specify how many splits to do in their direction.
    # design decision: if image pixels do not divide properly, the image will be enlarged to divide properly,
    # padded with pad_value.
    # Example: ([0, 1, 2] -> can't be divided into 2 equivalent parts -> [0, 1, 2, pad_value] -> can be divided into
    # 2 equivalent parts -> [0, 1], [2, pad_value])
    @staticmethod
    def __split_image_to_rectangular_bins(image, horizontal_bins, vertical_bins, pad_value):
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


# TODO: local test main, to be removed later.
# if __name__ == '__main__':
#     a = np.linspace(0, 24, 25).reshape([5, 5, ])
#     VisualizationMap(a, "", 5, 5)
