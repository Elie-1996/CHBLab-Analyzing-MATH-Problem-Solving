from warnings import warn
import pandas as pd
import msgpack


# This will allow for the data to be available globally
class Data:
    normalized_df = None
    denormalized_df = None


def load_input_data():
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

    # ###### update the data ######
    # TODO: When we know the x_scale and y_scale, we should use update_input_data to update Data.normalized
    update_input_data(x_scale=0, y_scale=0, normalized_df_=df)


def update_input_data(x_scale, y_scale, normalized_df_=None, denormalized_df_=None):
    if normalized_df_ is not None and denormalized_df_ is not None:
        warn("Attempted to normalize/denormalize data without a possible update (both normalized and denormalized "
             "data were passed)")
    elif normalized_df_ is not None:
        time_array = normalized_df_['Timestamp'].copy(deep=True)
        x_array = normalized_df_['RightX'].copy(deep=True)
        y_array = normalized_df_['RightY'].copy(deep=True)

        time_array = time_array - time_array[0]
        x_array = x_array * x_scale
        y_array = y_array * y_scale

        # update denormalized
        Data.denormalized_df = normalized_df_.copy(deep=True)
        Data.denormalized_df['Timestamp'] = time_array
        Data.denormalized_df['RightX'] = x_array
        Data.denormalized_df['RightY'] = y_array

        # update normalized
        Data.normalized_df = normalized_df_.copy(deep=True)
        Data.normalized_df['Timestamp'] = time_array

    elif denormalized_df_ is not None:
        time_array = denormalized_df_['Timestamp'].copy(deep=True)
        x_array = denormalized_df_['RightX'].copy(deep=True)
        y_array = denormalized_df_['RightY'].copy(deep=True)

        time_array = time_array - time_array[0]
        x_array = x_array / x_scale
        y_array = y_array / y_scale

        # update denormalized
        Data.normalized_df = denormalized_df_.copy(deep=True)
        Data.normalized_df['Timestamp'] = time_array
        Data.normalized_df['RightX'] = x_array
        Data.normalized_df['RightY'] = y_array

        # update normalized
        Data.denormalized_df = denormalized_df_.copy(deep=True)
        Data.denormalized_df['Timestamp'] = time_array

    else:
        warn("Attempted to normalize/denormalize data with no given input (neither normalized nor denormalized data "
             "were passed)")
