import pandas as pd
import msgpack
import Analysis
from Visualization import Visualization

# TODO The x,y we get from norm_pos are normalize, we should get the real coordinate by multiplying each of them by
#  the length and width of the screen size
""" for eli - "df" is our main data frame, "pupil_data" is a data frame that includes additional info about the pupil
            df["RightX"] is the x coordinates of the right eye
            df["LeftX"] is the y coordinates of the right eye
            df["Diameter"] is the size of the right pupil
            df["Timestampe"] is the current frame Time stamp"""

if __name__ == '__main__':
    # making the data frame for pupil data
    pldata_dir = './000/pupil.pldata'

    with open(pldata_dir, 'rb') as f:
        pupil_data = [[msgpack.unpackb(payload)['timestamp'],
                       msgpack.unpackb(payload)['diameter'] / 10,
                       msgpack.unpackb(payload)['diameter_3d'],
                       msgpack.unpackb(payload)['norm_pos'][0],
                       msgpack.unpackb(payload)['norm_pos'][1]]
                      for _, payload in msgpack.Unpacker(f)]

    # here we will create panda df and choose names for columns
    pupil_data = pd.DataFrame(pupil_data, columns=['Timestamp', 'Diameter', 'Diameter_3d', 'norm_x', 'norm_y'])

    # making the data frame for gaze data
    pldata_dir = './000/gaze.pldata'

    with open(pldata_dir, 'rb') as f:
        gaze_data = [[msgpack.unpackb(payload)['timestamp'],
                      msgpack.unpackb(payload)['norm_pos'][0],
                      msgpack.unpackb(payload)['norm_pos'][1]]
                     for _, payload in msgpack.Unpacker(f)]

    # here we will create the panda df and choose names for columns (I didn't check for all possible columns yet)
    df = pd.DataFrame(gaze_data, columns=['Timestamp', 'RightX', 'RightY'])
    df['Diameter'] = pupil_data['Diameter']

    # converting to numpy array for later use
    time_array = df['Timestamp'].to_numpy()
    x_array = df['RightX'].to_numpy()
    y_array = df['RightY'].to_numpy()

    # some visualization (heat map)
    Visualization.scatter_density(df)
    # finding fixations - more info about Sfix Efix in Analysis module
    Sfix, Efix = Analysis.fixation_detection(x_array, y_array, time_array)
    # find saccades - more info about Ssac Esac in Analysis module
    Ssac, Esac = Analysis.saccade_detection(x_array, y_array, time_array)

    print(df)
