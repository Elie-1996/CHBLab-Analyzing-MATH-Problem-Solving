import os
import pandas as pd
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
subjects_dict = {}

## Times per subject
question_3_002 = None
question_3_003 = (493, 613)
question_3_004 = (394, 560)
question_3_005 = (536, 649)
question_3_006 = None
question_3_007 = (468, 620)
question_3_008 = (720, 930)
question_3_009 = (584, 714)
question_3_010 = (431, 527)
question_3_011 = (385, 480)

subjects_dict['003_fixations'] = question_3_003
subjects_dict['004_fixations'] = question_3_004
subjects_dict['005_fixations'] = question_3_005
subjects_dict['007_fixations'] = question_3_007
subjects_dict['008_fixations'] = question_3_008
subjects_dict['009_fixations'] = question_3_008
subjects_dict['1000_fixations'] = question_3_008

data = []

input_directory = os.path.join('Subjects', 'data')

for file in os.listdir(input_directory):
    file_directory = os.path.join(input_directory, file)
    df = pd.read_csv(file_directory)
    print(file)
    current_subject = subjects_dict[file[:-4]]
    for i, point in enumerate(df.iterrows()):
        if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject[0] <= df['start_timestamp'].iloc[i] <= \
                df['start_timestamp'].iloc[0] + current_subject[1]:
            data.append([df['norm_pos_x'].iloc[i]*1810, (df['norm_pos_y'].iloc[i])*1014])

clustering = MeanShift().fit(data)

cluster_centers = clustering.cluster_centers_

# Finally We plot the data points
# and centroids in a 3D graph.
fig = plt.figure()
image_path = os.path.join('Heatmap', 'BackgroundImage.jpg')
img = Image.open(image_path)
img = ImageOps.flip(img)
ax = fig.add_subplot(111)
# ax.scatter(data[:, 0], data[:, 1], marker='o')
print(cluster_centers)
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', color='red', s=300, linewidth=5, zorder=10)
plt.imshow(img, origin='lower')
plt.show()
