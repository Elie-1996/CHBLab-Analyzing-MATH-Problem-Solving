import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy


def scatterDensity(df):
    x = df['RightX']
    y = df['RightY']

    ################# Plotting ####################################################
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


