import os
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import numpy as np
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
    min_i = 100000
    for i, point in enumerate(df.iterrows()):
        if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject[0] <= df['start_timestamp'].iloc[i] <= \
                df['start_timestamp'].iloc[0] + current_subject[1]:
            min_i = i
            duration = int(df['duration'].iloc[i]/10)
            # duration = int((df['start_timestamp'].iloc[i]-df['start_timestamp'].iloc[i-1])*1000)
            for j in range(duration):
                data.append([df['norm_pos_x'].iloc[i]*1810, (df['norm_pos_y'].iloc[i])*1014])

X = np.array(data)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=5000)

clustering = MeanShift(bandwidth=bandwidth, max_iter=300, n_jobs=2, bin_seeding=True).fit(X)
cluster_centers = clustering.cluster_centers_
labels = clustering.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Finish clustering")

# # Finally We plot the centroids
# fig = plt.figure()
# image_path = os.path.join('Heatmap', 'BackgroundImage.jpg')
# img = Image.open(image_path)
# img = ImageOps.flip(img)
# ax = fig.add_subplot(111)
# # ax.scatter(data[:, 0], data[:, 1], marker='o')
# print(cluster_centers)
# ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', color='red', s=300, linewidth=5, zorder=10)
# plt.imshow(img, origin='lower')
# plt.show()

# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

image_path = os.path.join('Heatmap', 'BackgroundImage.jpg')
img = Image.open(image_path)
img = ImageOps.flip(img)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = [i for i, x in enumerate(labels) if x == k]
    # my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.imshow(img, origin='lower')
plt.show()
