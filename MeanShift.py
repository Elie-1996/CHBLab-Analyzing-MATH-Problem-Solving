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


class rect:
    def __init__(self, upper_left_x, upper_left_y, bottom_right_x, bottom_right_y):
        self.x1 = upper_left_x
        self.y1 = upper_left_y
        self.x2 = bottom_right_x
        self.y2 = bottom_right_y

    def is_point_inside(self, _x, _y):
        # Note: y2 < y1 cuz the way we output the data (y axis is flipped)
        return self.x1 <= _x <= self.x2 and self.y2 <= _y <= self.y1


WIDTH = 1810
HEIGHT = 1014
rectangles_to_exclude = [rect(0.0, HEIGHT/2.0, WIDTH, 0.0)]
# rectangles_to_exclude = []  # keep this line if u want to include all points, otherwise comment it out

def should_exclude_point(_x, _y):
    for r in rectangles_to_exclude:
        if r.is_point_inside(x, y):
            return True
    return False


xy_points_amount_per_subject = [0]
for file in os.listdir(input_directory):
    file_directory = os.path.join(input_directory, file)
    df = pd.read_csv(file_directory)
    print(file)
    current_subject = subjects_dict[file[:-4]]
    min_i = 100000
    for i, point in enumerate(df.iterrows()):
        x, y = df['norm_pos_x'].iloc[i]*WIDTH, df['norm_pos_y'].iloc[i]*HEIGHT
        if should_exclude_point(x, y):
            continue
        if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject[0] <= df['start_timestamp'].iloc[i] <= \
                df['start_timestamp'].iloc[0] + current_subject[1]:
            min_i = i
            duration = int(df['duration'].iloc[i]/10)
            # duration = int((df['start_timestamp'].iloc[i]-df['start_timestamp'].iloc[i-1])*1000)
            for j in range(duration):
                data.append([df['norm_pos_x'].iloc[i]*WIDTH, (df['norm_pos_y'].iloc[i])*HEIGHT])
    xy_points_amount_per_subject.append(len(data) - xy_points_amount_per_subject[-1])
xy_points_amount_per_subject = xy_points_amount_per_subject[1:]

X = np.array(data)
print(X.shape)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=5000)
print("radius/bandwidth=" + str(bandwidth))

clustering = MeanShift(bandwidth=bandwidth, max_iter=300, n_jobs=2, bin_seeding=True).fit(X)
cluster_centers = clustering.cluster_centers_
labels = clustering.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Finish clustering")

# build 'n_clusters_'x'n_clusters_' switch_matrix
switch_mat = np.zeros([n_clusters_, n_clusters_])
start = 0
for a in xy_points_amount_per_subject:
    end = start + a
    subject_data = X[start:end, :]
    last_label = -1
    for idx, label in enumerate(labels):
        if last_label != -1:
            if idx >= subject_data.shape[0]:
                break
            x, y = subject_data[idx, 0], subject_data[idx, 1]
            switch_mat[last_label, label] += 1.0
        last_label = label

    start = end

from matrixHeatMap import heatmapMatrix, annotate_heatmapMatrix
for i in range(switch_mat.shape[0]):
        switch_mat[i][i] = 0
plt.figure(1)

im, cbar = heatmapMatrix(switch_mat, np.arange(n_clusters_), np.arange(n_clusters_),
                   cmap="YlGn", cbarlabel="Frequency")
texts = annotate_heatmapMatrix(im)


# Plot result
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle

plt.figure(2)
plt.clf()

hist = []
hist_color = []
image_path = os.path.join('Heatmap', 'BackgroundImage.jpg')
img = Image.open(image_path)
img = ImageOps.flip(img)
legend=[]
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = [i for i, x in enumerate(labels) if x == k]
    hist.append(len(my_members))
    hist_color.append(col)
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    patch = mpatches.Patch(color=col, label=k)
    legend.append(patch)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

plt.legend(handles=legend)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.imshow(img, origin='lower')


# turn to probablistic histogram
hist = [x/sum(hist) for x in hist]

# draw histogram
plt.figure(3)
plt.title("Cluster Histogram")
plt.xlabel("Cluster")
plt.ylabel("Count")

y_pos = np.arange(len(hist_color))
plt.bar(y_pos, hist, color=hist_color)
plt.xticks(y_pos, hist_color)
plt.show()

