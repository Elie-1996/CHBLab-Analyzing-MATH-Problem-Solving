import os
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KNeighborsClassifier as KNearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import numpy as np
from Utils import background_images, WIDTHS, HEIGHTS, subjects_dict, input_fixations_directory, input_pupil_directory

figure_counter = 0  # keep 0 please

DRAW_PUPIL_MEAN_HISTOGRAM = False
CREATE_CLUSTERS = False  # when true - clusters will be estimated from the data. when False, clusters will be loaded.
SAVE_DATA = True  # When true - saves all (x, y)
NEAREST_NEIGHBOR_K = 9
CLUSTER_RADIUS = [-1, -1, -1, -1]  # any non-negative number means this will use the fixed value given. If a negative value, then an automatic radius (bandwidth) estimation is performed

class Rect:
    def __init__(self, upper_left_x, upper_left_y, bottom_right_x, bottom_right_y):
        self.x1 = upper_left_x
        self.y1 = upper_left_y
        self.x2 = bottom_right_x
        self.y2 = bottom_right_y

    def is_point_inside(self, _x, _y):
        # Note: y2 < y1 cuz the way we output the data (y axis is flipped)
        return self.x1 <= _x <= self.x2 and self.y2 <= _y <= self.y1


# type in all the exculsions you would like
rectangles_to_exclude_question_1 = []
rectangles_to_exclude_question_2 = []
rectangles_to_exclude_question_3 = [Rect(0.0, HEIGHTS[2] / 2.0, WIDTHS[2], 0.0)]
rectangles_to_exclude_question_4 = []
rectangles_to_exclude = [
    rectangles_to_exclude_question_1,
    rectangles_to_exclude_question_2,
    rectangles_to_exclude_question_3,
    rectangles_to_exclude_question_4
]

# text colors - keep unchanged please:
CRED = '\033[91m'
CGREEN = '\33[32m'
CEND = '\033[0m'


def should_exclude_point(_x, _y, question_idx):
    for r in rectangles_to_exclude[question_idx]:
        if r.is_point_inside(_x, _y):
            return True
    return False


def choose_bandwidth(X, question_idx):
    bandwidth = CLUSTER_RADIUS[question_idx] if CLUSTER_RADIUS[question_idx] > 0 \
        else estimate_bandwidth(X, quantile=0.2, n_samples=5000)
    print("radius/bandwidth=" + str(bandwidth))

    return bandwidth


def get_question_fixation_data(question_idx):
    print("<Reading/Processing Fixation Data>:")

    data = []
    durations = {}
    WIDTH = WIDTHS[question_idx]
    HEIGHT = HEIGHTS[question_idx]
    xy_points_amount_per_subject = [0]
    subjects = []
    current_question_time_stamps = []
    for file in os.listdir(input_fixations_directory):
        file_directory = os.path.join(input_fixations_directory, file)
        df = pd.read_csv(file_directory)
        print(file)
        current_subject_times = subjects_dict[file.split("_")[0]]  # list of questions timelines
        current_subject_times = current_subject_times[
            question_idx]  # tuple of (start, end) timeline per question_idx
        if current_subject_times is None:
            continue

        for i, point in enumerate(df.iterrows()):
            x, y = df['norm_pos_x'].iloc[i] * WIDTH, df['norm_pos_y'].iloc[i] * HEIGHT
            if should_exclude_point(x, y, question_idx):
                continue
            if df['on_surf'].iloc[i] and df['start_timestamp'].iloc[0] + current_subject_times[0] <= \
                    df['start_timestamp'].iloc[i] <= \
                    df['start_timestamp'].iloc[0] + current_subject_times[1]:
                duration = int(df['duration'].iloc[i] / 10)
                durations[(x, y)] = df['duration'].iloc[i]
                for j in range(duration):
                    data.append([df['start_timestamp'].iloc[i], [df['norm_pos_x'].iloc[i] * WIDTH, (df['norm_pos_y'].iloc[i]) * HEIGHT]])
                    current_question_time_stamps.append(df['start_timestamp'])
        xy_points_amount_per_subject.append(len(data) - sum(xy_points_amount_per_subject))
        subjects.append(file.split('_')[0])
    xy_points_amount_per_subject = xy_points_amount_per_subject[1:]
    total_duration = np.array(list(durations.values())).sum()
    print("<Reading/Processing Fixation Data Complete!>")
    return data, xy_points_amount_per_subject, subjects, durations, total_duration


def get_question_pupil_data(question_idx):
    print("<Reading/Processing Pupil Data>:")

    diameter_data = []
    for file in os.listdir(input_pupil_directory):
        diameter_data_current_subject = []
        file_directory = os.path.join(input_pupil_directory, file)
        df = pd.read_csv(file_directory)
        print(file)
        current_subject_times = subjects_dict[file.split("_")[0]]  # list of questions timelines
        current_subject_times = current_subject_times[
            question_idx]  # tuple of (start, end) timeline per question_idx
        if current_subject_times is None:
            continue
        for i, point in enumerate(df.iterrows()):

            # only include diameter data within the given questions' timestamp
            if df['pupil_timestamp'].iloc[0] + current_subject_times[0] <= \
                    df['pupil_timestamp'].iloc[i] <= \
                    df['pupil_timestamp'].iloc[0] + current_subject_times[1]:
                diameter_time = [df['pupil_timestamp'].iloc[i], df['diameter'].iloc[i]]
                diameter_data_current_subject.append(diameter_time)
        diameter_data.append(diameter_data_current_subject)
    print("<Reading/Processing Pupil Data Complete!>")
    return diameter_data


def get_question_data(question_idx):
    print("####################################")
    print(f"Question {question_idx + 1}:")
    data, xy_points_amount_per_subject, subjects, durations, total_duration = get_question_fixation_data(question_idx)
    diameter_data = get_question_pupil_data(question_idx)

    return data, xy_points_amount_per_subject, subjects, diameter_data, durations, total_duration


def build_cluster_jump_matrix_between_different_clusters(xy_points_amount_per_subject, X, labels, n_clusters_):
    switch_mat_list = [np.zeros([n_clusters_, n_clusters_]) for _ in xy_points_amount_per_subject]
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

    return switch_mat_list


def build_cluster_jump_array_within_same_cluster(switch_mat_list, xy_points_amount_per_subject, question_idx, n_clusters_, subjects):
    global figure_counter
    from matrixHeatMap import heatmapMatrix, annotate_heatmapMatrix
    total_jumps_within_each_area = [0 for _ in range(switch_mat_list[0].shape[0])]
    for subject_idx in range(len(xy_points_amount_per_subject)):
        for i in range(switch_mat_list[subject_idx].shape[0]):
            total_jumps_within_each_area[i] += (switch_mat_list[subject_idx][i][i])
            switch_mat_list[subject_idx][i][i] = 0
        plt.figure(figure_counter + 1)
        figure_counter += 1

        im, cbar = heatmapMatrix(switch_mat_list[subject_idx], np.arange(n_clusters_), np.arange(n_clusters_),
                                 cmap="YlGn", cbarlabel="Frequency")
        plt.title(f"Question {question_idx + 1} - Subject {subjects[subject_idx]}")
        texts = annotate_heatmapMatrix(im)

    return switch_mat_list, total_jumps_within_each_area


def reorganize_pupil_data_per_area(diameter_data, XY_TIME, labels, n_clusters_):

    epsilon = 0.5
    diameters_in_all_clusters = [[] for _ in range(n_clusters_)]
    for current_subject_diameter_data in diameter_data:
        for t, diameter in current_subject_diameter_data:
            # find idx in XY_TIME
            idx = -1
            for (i, (search_time, (x, y))) in enumerate(XY_TIME):
                if -epsilon + t <= search_time <= epsilon + t:
                    idx = i
                    break
            if idx == -1:
                continue
            label = labels[idx]
            diameters_in_all_clusters[label].append(diameter)

    return diameters_in_all_clusters


def getAngle(three_angles):
    a, b, c = three_angles[0], three_angles[1], three_angles[2]
    import math
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def get_fixation_totalamount_variance_angles(XY_ONLY):
    angles = []
    for i in range(len(XY_ONLY) - 3):
        angles.append(getAngle(XY_ONLY[i:i+3]))
    angles = np.array(angles)
    angles_hist = []
    angle_degrees_strings = []
    degrees = 5
    cap = 30
    for i in range(0, cap, degrees):
        angles_hist.append(len([angle for angle in angles if i <= angle < i + degrees]))
        angle_degrees_strings.append(str(i) + "-" + str(i + degrees))
    angles_hist.append(len(angles) - sum(angles_hist))
    angle_degrees_strings.append(str(cap) + "+")
    return len(XY_ONLY), XY_ONLY.std(), angles.mean(), angles_hist, angle_degrees_strings


def get_pupil_mean_variance(diameter_data):
    flat_list = [item for sublist in diameter_data for item in sublist]
    return np.mean(flat_list), np.std(flat_list)


def get_duration_per_cluster(XY_ONLY, durations, labels, n_clusters_):
    durations_per_visit_per_cluster = [[] for cluster in range(n_clusters_)]
    old_x, old_y = -1, -1
    idx = -1
    total_stay = 0
    last_cluster = -1
    for x, y in XY_ONLY:
        idx += 1
        if x == old_x and y == old_y:
            continue
        old_x, old_y = x, y
        cluster = labels[idx]
        if cluster != last_cluster:
            if total_stay > 0:
                durations_per_visit_per_cluster[last_cluster].append(total_stay)
                total_stay = 0
            last_cluster = cluster
        duration = durations[(x, y)]
        total_stay += duration
    durations_per_visit_per_cluster[last_cluster].append(total_stay)


    stay_duration_mean_per_cluster = [np.array(duration_list_in_cluster).mean() for duration_list_in_cluster in durations_per_visit_per_cluster]
    return stay_duration_mean_per_cluster


def run_analysis():
    global figure_counter
    for question_idx in range(4):
        figure_counter = 0  # keep 0 please

        # extract question data
        data, xy_points_amount_per_subject, subjects, diameter_data, durations, total_duration = get_question_data(question_idx)
        if not data:  # if empty
            print(CRED + "Skipping - No data provided for current question" + CEND)
            continue
        XY_TIME = np.array(data)
        print(XY_TIME.shape)
        XY_ONLY = np.array([[x,y] for t,(x,y) in XY_TIME])
        print(XY_ONLY.shape)

        # choose bandwidth
        bandwidth = choose_bandwidth(XY_ONLY, question_idx)

        # create clusters and extract cluster data
        if CREATE_CLUSTERS:
            clustering = MeanShift(bandwidth=bandwidth, max_iter=300, n_jobs=2, bin_seeding=True).fit(XY_ONLY)
            cluster_centers = clustering.cluster_centers_
            labels = clustering.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            if SAVE_DATA:
                all_x = [x for x, y in XY_ONLY]
                all_y = [y for x, y in XY_ONLY]
                all_labels = labels
                df = pd.DataFrame(list(zip(all_x, all_y, all_labels)), columns=['x', 'y', 'labels'])
                df.to_csv('clusters.csv')
                df = pd.DataFrame(cluster_centers, columns=['cluster_centers_x', 'cluster_centers_y'])
                df.to_csv('cluster_centers.csv')
        else:
            KNN = KNearestNeighbors(n_neighbors=NEAREST_NEIGHBOR_K)
            df = pd.read_csv('clusters.csv')
            df_centers = pd.read_csv('cluster_centers.csv')
            loaded_data = [[df['x'].iloc[i], df['y'].iloc[i]] for i in range(len(df['x']))]
            loaded_labels = [df['labels'].iloc[i] for i in range(len(df['labels']))]
            cluster_centers = [[df_centers['cluster_centers_x'].iloc[i], df_centers['cluster_centers_y'].iloc[i]] for i in range(len(df_centers['cluster_centers_x']))]
            KNN.fit(loaded_data, loaded_labels)
            labels = KNN.predict(XY_ONLY)
            labels_unique = np.unique(loaded_labels)
            n_clusters_ = len(labels_unique)
        print(CGREEN + "Finish clustering" + CEND)

        print(xy_points_amount_per_subject)
        if DRAW_PUPIL_MEAN_HISTOGRAM:
            diameters_in_all_clusters = reorganize_pupil_data_per_area(diameter_data, XY_TIME, labels, n_clusters_)
        switch_mat_list = build_cluster_jump_matrix_between_different_clusters(xy_points_amount_per_subject, XY_ONLY, labels, n_clusters_)
        switch_mat_list, total_jumps_within_each_area = build_cluster_jump_array_within_same_cluster(switch_mat_list, xy_points_amount_per_subject, question_idx, n_clusters_, subjects)
        total_number_of_fixations, fixation_variance, angles_mean, angles_hist, angle_degrees_strings = get_fixation_totalamount_variance_angles(XY_ONLY)
        mean_pupil_size, variance_pupil_size = get_pupil_mean_variance(diameter_data)
        stay_duration_mean_per_cluster = get_duration_per_cluster(XY_ONLY, durations, labels, n_clusters_)

        # Plot result
        import matplotlib.patches as mpatches
        from itertools import cycle
        plt.figure(figure_counter + 1)
        figure_counter += 1
        plt.clf()

        hist = []
        hist_color = []
        image_path = os.path.join('Heatmap', background_images[question_idx])
        img = Image.open(image_path)
        img = ImageOps.flip(img)
        legend = []
        # colors = cycle('bgrcmykwbgrcmykbgrcmykbgrcmyk')
        colors = cycle(
            ['#CD6155', '#AF7AC5', '#2980B9', '#16A085', '#2ECC71', '#F1C40F', '#F39C12', '#ECF0F1', '#BDC3C7',
             '#95A5A6', '#707B7C', '#17202A'])
        # time_per_cluster = []
        for k, col in zip(range(n_clusters_), colors):
            my_members = [i for i, x in enumerate(labels) if x == k]
            # time_for_each_visit_in_current_cluster = get_every_visit_time(my_members, T)
            # time_per_cluster.append(time_for_each_visit_in_current_cluster)
            hist.append(len(my_members))
            hist_color.append(col)
            cluster_center = cluster_centers[k]
            plt.scatter(XY_ONLY[my_members, 0], XY_ONLY[my_members, 1], c=col, marker='.')
            patch = mpatches.Patch(color=col, label=k)
            legend.append(patch)
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

        plt.legend(handles=legend)
        plt.title(f'Question {question_idx + 1} - Estimated number of clusters: {n_clusters_}')
        plt.imshow(img, origin='lower')

        # turn to probablistic histogram
        hist = [x / sum(hist) for x in hist]

        # draw cluster histogram
        plt.figure(figure_counter + 1)
        figure_counter += 1
        plt.title(f"Question {question_idx + 1} - Cluster Histogram")
        plt.xlabel("Cluster")
        plt.ylabel("Count")

        y_pos = np.arange(len(hist_color))
        plt.bar(y_pos, hist, color=hist_color)
        plt.xticks(y_pos, np.arange(len(hist_color)))

        # draw jumps within each area histogram:
        plt.figure(figure_counter + 1)
        figure_counter += 1
        plt.title(f"Question {question_idx + 1} - Jumps per area Histogram")
        plt.xlabel("Area/Cluster")
        plt.ylabel("Average number of jumps")

        y_pos = np.arange(len(hist_color))
        average_jumps_within_each_area = [item / n_clusters_ for item in total_jumps_within_each_area]
        plt.bar(y_pos, average_jumps_within_each_area, color=hist_color)
        plt.xticks(y_pos, np.arange(len(hist_color)))

        # draw mean pupil histogram per cluster
        if DRAW_PUPIL_MEAN_HISTOGRAM:
            print("<Drawing Pupil Mean Histogram>")
            plt.figure(figure_counter + 1)
            figure_counter += 1
            plt.title(f"Question {question_idx + 1} - Mean Pupil Diameter")
            plt.xlabel("Area/Cluster")
            plt.ylabel("Mean Pupil Diameter")

            y_pos = np.arange(len(hist_color))
            from statistics import mean
            average_pupil_diameter = [mean(l1) if len(l1) > 0 else 0 for l1 in diameters_in_all_clusters]
            plt.bar(y_pos, average_pupil_diameter, color=hist_color)
            plt.xticks(y_pos, np.arange(len(hist_color)))
        else:
            print("<NOT! Drawing Pupil Mean Histogram>")

        # draw angles_hist
        plt.figure(figure_counter + 1)
        figure_counter += 1
        plt.title(f"Question {question_idx + 1} - Angles Hist")
        plt.xlabel("Degrees")
        plt.ylabel("Amount")
        y_pos = np.arange(len(angles_hist))
        plt.bar(y_pos, angles_hist)
        plt.xticks(y_pos, labels=angle_degrees_strings)

        # draw stay duration mean per cluster
        plt.figure(figure_counter + 1)
        figure_counter += 1
        plt.title(f"Question {question_idx + 1} - Duration mean per cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Mean Duration Stay (m/secs)")
        y_pos = np.arange(len(stay_duration_mean_per_cluster))
        plt.bar(y_pos, stay_duration_mean_per_cluster, color=hist_color)
        plt.xticks(y_pos, np.arange(len(stay_duration_mean_per_cluster)))

        # printing numbers:
        print("Length of Sequence: " + str(int(total_duration)) + " (m/secs)")
        print("Total Number Of Fixations: " + str(total_number_of_fixations))
        print("Fixation Variane: " + str(fixation_variance))
        print("Mean Pupil: " + str(mean_pupil_size))
        print("Variance Pupil: " + str(variance_pupil_size))
        print("Angles Mean: " + str(angles_mean))
        print([int(total_duration), int(total_number_of_fixations), int(fixation_variance), int(mean_pupil_size), int(variance_pupil_size), int(angles_mean)])
        print(CRED + "Please Close all open windows to continue to next question." + CEND)
        plt.show()


run_analysis()
print("####################################")
print(CGREEN + "Analysis presentations complete." + CEND)
