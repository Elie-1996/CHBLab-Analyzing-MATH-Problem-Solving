import math
import os

from PIL import Image
from heatmappy import Heatmapper
from heatmappy.video import VideoHeatmapper
import pandas as pd
import cv2

subjects_dict = {}
CRED = '\33[32m'
CEND = '\033[0m'

## Times per subject
question_3_002 = None
question_3_003 = (493, 613)
question_3_004 = (394, 560)
question_3_005 = (536, 649)
question_3_006 = None
question_3_007 = (468, 620)
question_3_008 = (720, 930)
question_3_009 = (584, 714)
# question_3_010 = (431, 527)
question_3_1000 = (316, 414)

subjects_dict['003'] = question_3_003
subjects_dict['004'] = question_3_004
subjects_dict['005'] = question_3_005
subjects_dict['007'] = question_3_007
subjects_dict['008'] = question_3_008
subjects_dict['009'] = question_3_009
subjects_dict['1000'] = question_3_1000

filename = 'BackgroundImage.jpg'
img = cv2.imread(filename)

################################################################
## For Video
# for subject in subjects_dict.keys():
#     print(CRED + "Creating heatmap for subject:", subject + CEND)
#
#     current_subject = subjects_dict[subject]
#     ## Creating video
#     img_array = []
#     filename = 'BackgroundImage.jpg'
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
#     for i in range(current_subject[1]-current_subject[0]):
#         out.write(img)
#     out.release()
#
#     print(CRED + "Video created" + CEND)
#
#     ## Creating heatmap
#     input_video = os.path.join('project.mp4')
#     input_points = os.path.join('data', subject+'_fixations.csv')
#
#     df = pd.read_csv(input_points)
#     coordinates = []
#     min_i = 100000
#     for i, point in enumerate(df.iterrows()):
#         if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject[0] <= df['start_timestamp'].iloc[i] <= \
#                 df['start_timestamp'].iloc[0] + current_subject[1]:
#             min_i = min(i, min_i)
#             coordinates.append((df['norm_pos_x'].iloc[i]*1810, (1-df['norm_pos_y'].iloc[i])*1014,
#                                 int((df['start_timestamp'].iloc[i]-df['start_timestamp'].iloc[min_i])*1000)))
#
#     img_heatmapper = Heatmapper()
#     video_heatmapper = VideoHeatmapper(img_heatmapper)
#
#     heatmap_video = video_heatmapper.heatmap_on_video_path(
#         video_path=input_video,
#         points=coordinates
#     )
#
#     heatmap_video.write_videofile('Accumulated_Subject-'+subject+'.mp4', bitrate="5000k", fps=24)
#     print(CRED + "Heatmap created" + CEND)

################################################################################################
##  For image
for subject in subjects_dict.keys():
    filename = 'BackgroundImage.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    img = Image.open(filename)
    heatmapper = Heatmapper()
    img_list = []
    input_points = os.path.join('data', subject + '_fixations.csv')
    current_subject = subjects_dict[subject]
    df = pd.read_csv(input_points)
    interval = 3    # Number of seconds
    num_rows = len(df)
    normalize_time = df['start_timestamp'].iloc[0]
    df['start_timestamp'] -= normalize_time
    idx = 0
    while True:
        if idx >= num_rows or df['start_timestamp'].iloc[idx] > current_subject[1]:
            break
        if df['on_surf'].iloc[idx] and current_subject[0] <= df['start_timestamp'].iloc[idx]:
            start_time = df['start_timestamp'].iloc[idx]
            current_points = []
            while True:
                if idx < num_rows and df['start_timestamp'].iloc[idx] - start_time <= interval:
                    if df['on_surf'].iloc[idx] and current_subject[0] <= df['start_timestamp'].iloc[idx] <= current_subject[1]:
                        current_points.append((df['norm_pos_x'].iloc[idx] * size[0], (1 - df['norm_pos_y'].iloc[idx])*size[1]))
                    idx += 1
                else:
                    break
            img_list.append(heatmapper.heatmap_on_img(current_points, img))
        idx += 1

    out = cv2.VideoWriter('Accumulated_Subject-' + subject + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

    for i in range(len(img_list)):
        img_list[i].save('heatmap.png')
        for j in range(5):
            img = cv2.imread('heatmap.png')
        out.write(img)
    out.release()
