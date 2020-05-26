from pathlib import Path
from warnings import warn
import pandas as pd
import msgpack
from . import Preprocessing


# Before relying on this class whatsoever, it is absolutely necessary to call DataLib.set_data or, preferably by calling
# This class will allow for the data to be available globally
class Data:
    x_resolution = 0
    y_resolution = 0

    pupil_data = None
    pupil_data_normalized = None

    gaze_data = None
    gaze_data_normalized = None

    fixation_data = None
    fixation_data_normalized = None

    blinks_data = None

    @staticmethod
    def change_resolution(x_res, y_res, update_resolution_parameters_only=False):
        Data.x_resolution = x_res
        Data.y_resolution = y_res

        # update the class, if that's what the user requests.
        if not update_resolution_parameters_only:
            Data.set_data(
                Data.pupil_data_normalized,
                Data.gaze_data_normalized,
                Data.fixation_data_normalized,
                Data.blinks_data,
                x_res=x_res,
                y_res=y_res
            )

    # =================================================================================
    # =============== helper methods (not to be used outside class) ===================
    @staticmethod
    def set_pupil_data(pupil_data, x_scale, y_scale):
        # start counting time from 0 for both normalized and regular data.
        time_array = pupil_data['Timestamp']
        time_array = time_array - time_array[0]
        time_array_normalized = time_array.copy(deep=True)

        x_array_normalized = pupil_data['X']
        x_array = x_array_normalized * x_scale
        y_array_normalized = pupil_data['Y']
        y_array = y_array_normalized * y_scale

        diameter_normalized = pupil_data['Diameter']
        diameter = pupil_data['Diameter']  # TODO: This is wrong, need to use Daniel's normalization method.
        diameter_3d_normalized = pupil_data['Diameter_3d']
        diameter_3d = pupil_data['Diameter_3d']  # TODO: This is wrong, need to use Daniel's normalization method.

        # Update all pupil data in the DataLib class.
        Data.pupil_data = pupil_data.copy(deep=True)
        Data.pupil_data['Timestamp'] = time_array
        Data.pupil_data['X'] = x_array
        Data.pupil_data['Y'] = y_array
        Data.pupil_data['Diameter'] = diameter
        Data.pupil_data['Diameter_3d'] = diameter_3d

        Data.pupil_data_normalized = pupil_data.copy(deep=True)
        Data.pupil_data_normalized['Timestamp'] = time_array_normalized
        Data.pupil_data_normalized['X'] = x_array_normalized
        Data.pupil_data_normalized['Y'] = y_array_normalized
        Data.pupil_data_normalized['Diameter'] = diameter_normalized
        Data.pupil_data_normalized['Diameter_3d'] = diameter_3d_normalized

    @staticmethod
    def set_gaze_data(gaze_data, x_scale, y_scale):
        # start counting time from 0 for both normalized and regular data.
        time_array = gaze_data['Timestamp']
        print(gaze_data)
        time_array = time_array - time_array[0]
        time_array_normalized = time_array.copy(deep=True)

        x_array_normalized = gaze_data['X']
        x_array = x_array_normalized * x_scale
        y_array_normalized = gaze_data['Y']
        y_array = y_array_normalized * y_scale

        # Update all gaze data in the DataLib class.
        Data.gaze_data = gaze_data.copy(deep=True)
        Data.gaze_data['Timestamp'] = time_array
        Data.gaze_data['X'] = x_array
        Data.gaze_data['Y'] = y_array

        Data.gaze_data_normalized = gaze_data.copy(deep=True)
        Data.gaze_data_normalized['Timestamp'] = time_array_normalized
        Data.gaze_data_normalized['X'] = x_array_normalized
        Data.gaze_data_normalized['Y'] = y_array_normalized

    @staticmethod
    def set_fixation_data(fixation_data, x_scale, y_scale):
        # start counting time from 0 for both normalized and regular data.
        time_array = fixation_data['Timestamp']
        time_array = time_array - time_array[0]
        time_array_normalized = time_array.copy(deep=True)

        x_array_normalized = fixation_data['X']
        x_array = x_array_normalized * x_scale
        y_array_normalized = fixation_data['Y']
        y_array = y_array_normalized * y_scale

        duration_array = fixation_data['Duration']
        duration_array_normalized = fixation_data['Duration']

        # Update all fixation data in the DataLib class.
        Data.fixation_data = fixation_data.copy(deep=True)
        Data.fixation_data['Timestamp'] = time_array
        Data.fixation_data['X'] = x_array
        Data.fixation_data['Y'] = y_array
        Data.fixation_data['Duration'] = duration_array

        Data.fixation_data_normalized = fixation_data.copy(deep=True)
        Data.fixation_data_normalized['Timestamp'] = time_array_normalized
        Data.fixation_data_normalized['X'] = x_array_normalized
        Data.fixation_data_normalized['Y'] = y_array_normalized
        Data.fixation_data_normalized['Duration'] = duration_array_normalized

    @staticmethod
    def set_blinks_data(blinks_data, x_scale, y_scale):
        # start counting time from 0 for both normalized and regular data.
        start_time_array = blinks_data['start_timestamp']
        time_shift = start_time_array[0]

        # set time data
        start_time_array = start_time_array - time_shift

        end_time_array = blinks_data['end_timestamp']
        end_time_array = end_time_array - time_shift

        duration_array = blinks_data['duration']

        start_frame_index_array = blinks_data['start_frame_index']
        end_frame_index_array = blinks_data['end_frame_index']
        index_array = blinks_data['index']

        # Update all blinks data in the DataLib class.
        Data.blinks_data = blinks_data.copy(deep=True)
        Data.blinks_data['start_timestamp'] = start_time_array
        Data.blinks_data['end_timestamp'] = end_time_array
        Data.blinks_data['Duration'] = duration_array
        Data.blinks_data['start_frame_index'] = start_frame_index_array
        Data.blinks_data['end_frame_index'] = end_frame_index_array
        Data.blinks_data['index'] = index_array

    # =================================================================================
    # ======================= setter and getter methods ===============================
    @staticmethod
    def set_data(pupil, gaze, fixation, blinks, x_res=None, y_res=None):
        """
        :param pupil: the data as received from the output of pupil player
        :param gaze: the data as received from the output of pupil player
        :param fixation: the data as received from the output of pupil player
        :param blinks: the data as received from the output of pupil player
        :param x_res: the resolution of which to scale x
        :param y_res: the resolution of which to scale y

        :return : sets all DataLib parameters as needed.
        """

        if x_res is None:
            x_res = Data.x_resolution
        if y_res is None:
            y_res = Data.y_resolution

        Data.change_resolution(x_res, y_res, update_resolution_parameters_only=True)

        # make sure we are in legal state.
        if x_res is None or x_res <= 0 or y_res is None or y_res <= 0:
            raise Exception("Raw data has incorrect scaled entered: either None or non-positive.")

        if pupil is not None:
            Data.set_pupil_data(pupil, x_res, y_res)
        if gaze is not None:
            Data.set_gaze_data(gaze, x_res, y_res)
        if fixation is not None:
            Data.set_fixation_data(fixation, x_res, y_res)
        if blinks is not None:
            Data.set_blinks_data(blinks, x_res, y_res)
        pass

    @staticmethod
    def read_only_pupil_data(get_normalized=False):
        if get_normalized:
            return Data.pupil_data_normalized
        return Data.pupil_data

    @staticmethod
    def read_only_gaze_data(get_normalized=False):
        if get_normalized:
            return Data.gaze_data_normalized
        return Data.gaze_data

    @staticmethod
    def read_only_fixation_data(get_normalized=False):
        if get_normalized:
            return Data.fixation_data_normalized
        return Data.fixation_data

    @staticmethod
    def preprocess():
        Preprocessing.filter_out_exceeding_gazes()
        Preprocessing.pupil_preprocessing()


def load_pupil_data(pldata_dir):
    # making the data frame for pupil data
    with open(pldata_dir, 'rb') as f:
        pupil_data = [[msgpack.unpackb(payload)[b'timestamp'],
                       msgpack.unpackb(payload)[b'diameter'] / 10,
                       msgpack.unpackb(payload)[b'diameter_3d'],
                       msgpack.unpackb(payload)[b'norm_pos'][0],
                       msgpack.unpackb(payload)[b'norm_pos'][1]]
                      for _, payload in msgpack.Unpacker(f)]

    # here we will create panda df and choose names for columns
    pupil_data_frame = pd.DataFrame(pupil_data, columns=['Timestamp', 'Diameter', 'Diameter_3d', 'X', 'Y'])
    return pupil_data_frame


def load_gaze_data(gazedata_dir):
    # making the data frame for gaze data
    gaze_data_frame = pd.read_csv(gazedata_dir)
    gaze_data_frame = gaze_data_frame[['world_timestamp', 'x_norm', 'y_norm']]
    gaze_data_frame.columns = ['Timestamp', 'X', 'Y']
    # here we will create the panda df and choose names for columns (I didn't check for all possible columns yet)
    return gaze_data_frame


def load_fixation_data(surface_fixation_dir):
    fixation_data_frame = pd.read_csv(surface_fixation_dir)
    fixation_data_frame = fixation_data_frame[['world_timestamp', 'norm_pos_x', 'norm_pos_y', 'duration']]
    fixation_data_frame.columns = ['Timestamp', 'X', 'Y', 'Duration']
    return fixation_data_frame


def load_input_data(x_res, y_res,
                    subject='000',
                    pupildata_dir='pupil.pldata',
                    gazedata_dir='gaze_positions_on_surface_Surface 1.csv',
                    fixationdata_dir='fixations_on_surface_Surface 1.csv',
                    blinksdata_dir='blinks.csv',
                    ):
    pupil_data, gaze_data, fixation_data, blinks_data = None, None, None, None

    if pupildata_dir is not None:
        pupil_path = Path(subject, pupildata_dir)
        pupil_data = load_pupil_data(pupil_path)

    if gazedata_dir is not None:
        gaze_path = Path(subject, 'exports', '000', 'surfaces', gazedata_dir)
        gaze_data = load_gaze_data(gaze_path)

    if fixationdata_dir is not None:
        fixations_path = Path(subject, 'exports', '000', 'surfaces', fixationdata_dir)
        fixation_data = load_fixation_data(fixations_path)

    if blinksdata_dir is not None:
        blinks_path = Path(subject, 'exports', '000', blinksdata_dir)
        blinks_data = load_blinks_data(blinks_path)

    Data.set_data(pupil_data, gaze_data, fixation_data, blinks_data, x_res=x_res, y_res=y_res)

    Data.preprocess()


def load_blinks_data(path):
    blinks_df = pd.read_csv(path)
    return blinks_df



