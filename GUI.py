from tkinter import *

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from Visualization import VisualizationMap
import numpy as np
from warnings import warn


def setup_gui(vm: VisualizationMap):
    m = Tk()
    m.title('HeatMap')

    menu_button_frame = Frame()
    menu_buttons_widget(m, menu_button_frame)

    map_frame = Frame()
    hm_ax, hm_canvas = heatmap_widget(map_frame, vm)

    widgets_frame = Frame()
    time_widget(widgets_frame, vm, hm_ax, hm_canvas)
    bin_adjustment_widget(widgets_frame, vm, hm_ax, hm_canvas)

    menu_button_frame.pack()
    map_frame.pack()
    widgets_frame.pack()

    m.mainloop()


def menu_buttons_widget(parent, m):
    mb = Menubutton(m, text="File")
    mb.grid()
    mb.menu = Menu(mb, tearoff=0)
    mb["menu"] = mb.menu
    cVar = IntVar()
    aVar = IntVar()
    mb.menu.add_checkbutton(label='Load...', variable=cVar)
    mb.menu.add_checkbutton(label='Exit', command=parent.destroy, variable=aVar)
    mb.pack()


def compute_relevant_coordinates(vm):
    x_coords, y_coords = [], []
    for current_bin in vm.bins:
        for (x, y) in current_bin:
            x_coords.append(x)
            y_coords.append(y)
    return x_coords, y_coords


def heatmap_widget(m, vm):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    f = plt.Figure(figsize=(5, 5), dpi=100)
    ax = f.add_subplot(111)
    canvas = FigureCanvasTkAgg(f, m)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
    substitute_heatmap_plot(vm, ax, canvas)
    return ax, canvas


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics

    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    :param facecolor: 
    :param x : array_like, shape (n, )
        Input data.
    :param y : array_like, shape (n, )
        Input data.
    :param ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    :param n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    # render plot with "plt.show()".


def draw_ellipses(vm, ax):
    for current_bin in vm.bins:
        bin_len = len(current_bin)
        all_x = np.array([current_bin[i][0] for i in range(bin_len)])
        all_y = np.array([current_bin[i][1] for i in range(bin_len)])

        if len(all_x) > 1:
            # TODO: Accuracy of the ellipse should also be controlled from GUI!
            e_g = confidence_ellipse(all_x, all_y, ax, 1.5)
            e_b = confidence_ellipse(all_x, all_y, ax, 1.0)
            e_r = confidence_ellipse(all_x, all_y, ax, 0.5)

            # draw ellipse
            ax.add_artist(e_g)
            e_g.set_clip_box(ax.bbox)
            e_g.set_alpha(0.1)
            e_g.set_facecolor([0.0, 0.99, 0.0])

            ax.add_artist(e_g)
            e_b.set_clip_box(ax.bbox)
            e_b.set_alpha(0.5)
            e_b.set_facecolor([0.0, 0.0, 0.99])

            ax.add_artist(e_g)
            e_r.set_clip_box(ax.bbox)
            e_r.set_alpha(0.8)
            e_r.set_facecolor([0.99, 0.0, 0.0])


def draw_points(ax, x_list, y_list, s):
    ax.scatter(x_list, y_list, s=s, color=[255/256, 255/256, 255/256])


def substitute_heatmap_plot(vm, ax, canvas):
    # x_coords, y_coords = compute_relevant_coordinates(vm)
    ax.clear()
    ax.imshow(vm.full_image)
    x_coords, y_coords = compute_relevant_coordinates(vm)
    draw_points(ax, x_coords, y_coords, s=12)
    draw_ellipses(vm, ax)

    canvas.draw()


def attempt_to_update_timeframe(vm, ax, canvas, start_time_str, end_time_str):
    try:
        start = float(start_time_str)
        end = float(end_time_str)
    except ValueError:
        warn("Could not parse interval correctly: at least start or end interval was incorrectly entered")
        return
    vm.update_interval(start, end)
    substitute_heatmap_plot(vm, ax, canvas)


def attempt_to_update_division(vm, ax, canvas, horizontal_division, vertical_division):
    try:
        horizontal = int(horizontal_division)
        vertical = int(vertical_division)
    except ValueError:
        warn("Could not parse interval correctly: at least horizontal or vertical division was incorrectly entered")
        return
    vm.update_bin_division(horizontal, vertical)
    substitute_heatmap_plot(vm, ax, canvas)


def time_widget(m, vm, ax, canvas):
    Label(m, text="Choose Timeframe (in seconds):").grid(row=1)

    Label(m, text='From ').grid(row=2)
    Label(m, text='To: ').grid(row=3)
    start_time_interval_entry = Entry(m)
    start_time_interval_entry.grid(row=2, column=1)
    end_time_interval_entry = Entry(m)
    end_time_interval_entry.grid(row=3, column=1)
    confirmButton = Button(m, text="Confirm", width=16, command=lambda: attempt_to_update_timeframe(
        vm,
        ax,
        canvas,
        start_time_interval_entry.get(),
        end_time_interval_entry.get()
    ))
    confirmButton.grid(row=4, column=1)


def bin_adjustment_widget(m, vm, ax, canvas):
    Label(m, text="Choose Grid Division:").grid(row=1, column=2)

    Label(m, text="Horizontal (->):").grid(row=2, column=2)
    Label(m, text="Vertical (^):").grid(row=3, column=2)
    horizontal_entry = Entry(m)
    horizontal_entry.grid(row=2, column=3)
    vertical_entry = Entry(m)
    vertical_entry.grid(row=3, column=3)
    confirmButton = Button(m, text="Confirm", width=16, command=lambda: attempt_to_update_division(
        vm,
        ax,
        canvas,
        horizontal_entry.get(),
        vertical_entry.get()
    ))
    confirmButton.grid(row=4, column=3)
