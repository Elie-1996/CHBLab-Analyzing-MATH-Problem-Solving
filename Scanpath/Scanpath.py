import os
import random

import cv2
import numpy as np
import _get_data_functions as get
import pandas as pd


def scanpath(animation=True, wait_time=30000, putLines=True, putNumbers=False, plotMaxDim=1024):
    ''' This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.
        By default, one random scanpath is chosen between available subjects. For
        a specific subject, it is possible to specify its id on the additional
        argument subject=id.
        It is possible to visualize it as an animation by setting the additional
        argument animation=True.
        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''
    ## Loading Data
    img_path = os.path.join('.', 'Heatmap', 'BackgroundImage.jpg')
    subject_path = os.path.join('.', 'Subjects', 'data', '003_fixations.csv')
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height)
    df = pd.read_csv(subject_path)

    ## Init vars
    scanpath = []
    subjects_dict = {}
    question_3_003 = (493, 613)
    subjects_dict['003'] = question_3_003

    ## Preprocessing
    normalize_time = df['start_timestamp'].iloc[0]
    df['start_timestamp'] -= normalize_time
    df['norm_pos_x'] = df[df['norm_pos_x'] >= 0]['norm_pos_x']
    df['norm_pos_x'] = df[df['norm_pos_x'] <= 1]['norm_pos_x']
    df['norm_pos_y'] = df[df['norm_pos_y'] >= 0]['norm_pos_y']
    df['norm_pos_y'] = df[df['norm_pos_y'] <= 1]['norm_pos_y']

    df['norm_pos_x'] = pd.cut(df['norm_pos_x'], 9)
    df['norm_pos_y'] = pd.cut(df['norm_pos_y'], 9)
    current_subject = subjects_dict['003']
    num_rows = len(df)

    print("Preprocess data")
    idx = 0
    while True:
        if idx >= num_rows or df['start_timestamp'].iloc[idx] > current_subject[1]:
            break
        if df['on_surf'].iloc[idx] and current_subject[0] <= df['start_timestamp'].iloc[idx]:
            scanpath.append([df['norm_pos_x'].iloc[idx].mid * size[0], (1 - df['norm_pos_y'].iloc[idx].mid) * size[1],
                             df['start_timestamp'].iloc[idx],
                             df['start_timestamp'].iloc[idx + 1] - df['start_timestamp'].iloc[idx]])
        idx += 1

    toPlot = [img, ]  # look, it is a list!
    scanpath = np.asarray(scanpath)
    left_color = [0, 0, 0]
    right_color = [0, 0, 0]

    ## Creating scanpath
    print("Fixations are ready, start making the scanpath")
    left_ind = 0
    right_ind = 1
    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        cv2.circle(frame, (fixation[0], fixation[1]), 10, (0, 204, 0), -1)
        if putNumbers:
            cv2.putText(frame, str(i + 1), (fixation[0], fixation[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        thickness=2)
        if putLines and i > 0:
            prec_fixation = scanpath[i - 1].astype(int)
            new_color = (i * 4) % 256

            ## For arrow ->
            if prec_fixation[0] > fixation[0] or prec_fixation[1] > fixation[1]:
                left_color[left_ind] = new_color
                left_color[right_ind] = 0
                cv2.arrowedLine(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                                left_color, thickness=3, shift=0)
            ## For arrow <-
            else:
                right_color[right_ind] = new_color
                right_color[left_ind] = 0
                cv2.arrowedLine(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                                right_color, thickness=3, shift=0)

            ## Initialize new color
            if new_color == 0:
                color1 = random.randint(128, 256)
                left_color = [0, 0, 0]
                right_color = [0, 0, 0]
                left_ind = 0
                right_ind = 0
                while left_ind == right_ind:
                    left_ind = random.randint(0, 2)
                    right_ind = random.randint(0, 2)

                for index in range(1, 3):
                    if index != left_ind and index != right_ind:
                        left_color[index] = color1

                left_color[left_ind] = 128

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation:
            toPlot.pop(0)

    # if required, resize the frames
    if plotMaxDim:
        for i in range(len(toPlot)):
            h, w, _ = np.shape(toPlot[i])
            h, w = float(h), float(w)
            if h > w:
                w = (plotMaxDim / h) * w
                h = plotMaxDim
            else:
                h = (plotMaxDim / w) * h
                w = plotMaxDim
            h, w = int(h), int(w)
            toPlot[i] = cv2.resize(toPlot[i], (w, h), interpolation=cv2.INTER_CUBIC)

    print("Now its imshow")
    for i in range(len(toPlot)):
        cv2.imshow('Scanpath of ' + '003' + ' watching ' + 'Question 3',
                   toPlot[i])
        if i == 0:
            milliseconds = 1
        elif i == 1:
            milliseconds = scanpath[0, 3]
        else:
            milliseconds = scanpath[i - 1, 3] - scanpath[i - 2, 2]
        milliseconds *= 1000

        cv2.waitKey(250)

    # cv2.waitKey(wait_time)
    #
    cv2.destroyAllWindows()

    print("Finish video")


scanpath()
