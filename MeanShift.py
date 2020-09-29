import os
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import numpy as np
from Utils import background_images, WIDTHS, HEIGHTS, subjects_dict, input_fixations_directory

CLUSTER_RADIUS = [-1, 200, -1, -1]  # any non-negative number means this will use the fixed value given. If a negative value, then an automatic radius (bandwidth) estimation is performed


class rect:
    def __init__(self, upper_left_x, upper_left_y, bottom_right_x, bottom_right_y):
        self.x1 = upper_left_x
        self.y1 = upper_left_y
        self.x2 = bottom_right_x
        self.y2 = bottom_right_y

    def is_point_inside(self, _x, _y):
        # Note: y2 < y1 cuz the way we output the data (y axis is flipped)
        return self.x1 <= _x <= self.x2 and self.y2 <= _y <= self.y1


rectangles_to_exclude_question_1 = []
rectangles_to_exclude_question_2 = []
rectangles_to_exclude_question_3 = [rect(0.0, HEIGHTS[2]/2.0, WIDTHS[2], 0.0)]
rectangles_to_exclude_question_4 = []
rectangles_to_exclude = [
    rectangles_to_exclude_question_1,
    rectangles_to_exclude_question_2,
    rectangles_to_exclude_question_3,
    rectangles_to_exclude_question_4
]


def should_exclude_point(_x, _y, question_idx):
    for r in rectangles_to_exclude[question_idx]:
        if r.is_point_inside(x, y):
            return True
    return False


for question_idx in range(4):
    print("####################################")
    print(f"Question {question_idx + 1}:")
    data = []
    WIDTH = WIDTHS[question_idx]
    HEIGHT = HEIGHTS[question_idx]
    xy_points_amount_per_subject = [0]
    subjects = []
    for file in os.listdir(input_fixations_directory):
        file_directory = os.path.join(input_fixations_directory, file)
        df = pd.read_csv(file_directory)
        print(file)
        current_subject_times = subjects_dict[file[:-4]]  # list of questions timelines
        current_subject_times = current_subject_times[question_idx]  # tuple of (start, end) timeline per question_idx
        if current_subject_times is None:
            continue
        for i, point in enumerate(df.iterrows()):
            x, y = df['norm_pos_x'].iloc[i]*WIDTH, df['norm_pos_y'].iloc[i]*HEIGHT
            if should_exclude_point(x, y, question_idx):
                continue
            if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject_times[0] <= df['start_timestamp'].iloc[i] <= \
                    df['start_timestamp'].iloc[0] + current_subject_times[1]:
                duration = int(df['duration'].iloc[i]/10)
                # duration = int((df['start_timestamp'].iloc[i]-df['start_timestamp'].iloc[i-1])*1000)
                for j in range(duration):
                    data.append([df['norm_pos_x'].iloc[i]*WIDTH, (df['norm_pos_y'].iloc[i])*HEIGHT])
        xy_points_amount_per_subject.append(len(data) - sum(xy_points_amount_per_subject))
        subjects.append(file.split('_')[0])
    xy_points_amount_per_subject = xy_points_amount_per_subject[1:]

    if not data:  # if empty
        print("Skipping - No data provided for current question")
        continue
    X = np.array(data)
    print(X.shape)
    bandwidth = CLUSTER_RADIUS[question_idx] if CLUSTER_RADIUS[question_idx] > 0 else estimate_bandwidth(X, quantile=0.2, n_samples=5000)
    print("radius/bandwidth=" + str(bandwidth))

    clustering = MeanShift(bandwidth=bandwidth, max_iter=300, n_jobs=2, bin_seeding=True).fit(X)
    cluster_centers = clustering.cluster_centers_
    labels = clustering.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    CRED = '\033[91m'
    CGREEN = '\33[32m'
    CEND = '\033[0m'
    print(CGREEN + "Finish clustering" + CEND)
    print(CRED + "Please Close all open windows to continue to next question." + CEND)

    # build 'n_clusters_'x'n_clusters_' switch_matrix
    switch_mat_list = [np.zeros([n_clusters_, n_clusters_]) for _ in xy_points_amount_per_subject]
    print(xy_points_amount_per_subject)

    start = 0
    for subject_idx, a in enumerate(xy_points_amount_per_subject):
        end = start + a
        subject_data = X[start:end, :]
        last_label = -1
        for idx, label in enumerate(labels):
            if last_label != -1:
                if idx >= subject_data.shape[0]:
                    break
                x, y = subject_data[idx, 0], subject_data[idx, 1]
                switch_mat_list[subject_idx][last_label, label] += 1.0
            last_label = label

        start = end

    from matrixHeatMap import heatmapMatrix, annotate_heatmapMatrix
    for subject_idx in range(len(xy_points_amount_per_subject)):
        for i in range(switch_mat_list[subject_idx].shape[0]):
                switch_mat_list[subject_idx][i][i] = 0
        plt.figure(subject_idx+3)

        im, cbar = heatmapMatrix(switch_mat_list[subject_idx], np.arange(n_clusters_), np.arange(n_clusters_),
                           cmap="YlGn", cbarlabel="Frequency")
        plt.title(f"Question {question_idx + 1} - Subject {subjects[subject_idx]}")
        texts = annotate_heatmapMatrix(im)


    # Plot result
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    hist = []
    hist_color = []
    image_path = os.path.join('Heatmap', background_images[question_idx])
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
    plt.title(f'Question {question_idx + 1} - Estimated number of clusters: {n_clusters_}')
    plt.imshow(img, origin='lower')


    # turn to probablistic histogram
    hist = [x/sum(hist) for x in hist]

    # draw histogram
    plt.figure(2)
    plt.title(f"Question {question_idx + 1} - Cluster Histogram")
    plt.xlabel("Cluster")
    plt.ylabel("Count")

    y_pos = np.arange(len(hist_color))
    plt.bar(y_pos, hist, color=hist_color)
    plt.xticks(y_pos, hist_color)
    plt.show()


