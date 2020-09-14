import os
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
# subjects_dict['004'] = question_3_004
# subjects_dict['005'] = question_3_005
# subjects_dict['007'] = question_3_007
# subjects_dict['008'] = question_3_008
# subjects_dict['009'] = question_3_009
# subjects_dict['1000'] = question_3_1000

for subject in subjects_dict.keys():
    print(CRED + "Creating heatmap for subject:", subject + CEND)
    current_subject = subjects_dict[subject]
    ## Creating video
    img_array = []
    filename = 'BackgroundImage.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
    for i in range(current_subject[1]-current_subject[0]):
        out.write(img)
    out.release()

    print(CRED + "Video created" + CEND)

    ## Creating heatmap
    input_video = os.path.join('project.mp4')
    input_points = os.path.join('data', subject+'_fixations.csv')

    df = pd.read_csv(input_points)
    coordinates = []
    min_i = 100000
    for i, point in enumerate(df.iterrows()):
        if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject[0] <= df['start_timestamp'].iloc[i] <= \
                df['start_timestamp'].iloc[0] + current_subject[1]:
            min_i = min(i, min_i)
            coordinates.append((df['norm_pos_x'].iloc[i]*1810, (1-df['norm_pos_y'].iloc[i])*1014,
                                int((df['start_timestamp'].iloc[i]-df['start_timestamp'].iloc[min_i])*1000)))

    img_heatmapper = Heatmapper()
    video_heatmapper = VideoHeatmapper(img_heatmapper)

    heatmap_video = video_heatmapper.heatmap_on_video_path(
        video_path=input_video,
        points=coordinates
    )

    heatmap_video.write_videofile('Subject-'+subject+'.mp4', bitrate="5000k", fps=24)
    print(CRED + "Heatmap created" + CEND)

