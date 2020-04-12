from tkinter import *
from Visualization import VisualizationMap
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

    # representation should change but this is a good start for the testing for now :)

    x_coords, y_coords = compute_relevant_coordinates(vm)

    f = plt.Figure(figsize=(5, 5), dpi=100)
    ax = f.add_subplot(111)
    canvas = FigureCanvasTkAgg(f, m)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
    substitute_plot(vm, ax, canvas, x_coords, y_coords)
    return ax, canvas


def substitute_plot(vm, ax, canvas, x_coords, y_coords):
    print("------------------------------")
    print(x_coords)
    print(y_coords)
    ax.clear()
    ax.imshow(vm.full_image)
    ax.plot(x_coords, y_coords)
    canvas.draw()


def attempt_update(vm, ax, canvas, start_time_str, end_time_str):
    try:
        start = float(start_time_str)
        end = float(end_time_str)
    except ValueError:
        warn("Could not parse interval correctly: at least start or end interval was incorrectly entered")
        return
    vm.update_interval(start, end)
    x_coords, y_coords = compute_relevant_coordinates(vm)
    substitute_plot(vm, ax, canvas, x_coords, y_coords)


def time_widget(m, vm, ax, canvas):
    Label(m, text="Choose Timeframe (in seconds):").grid(row=1)

    Label(m, text='From ').grid(row=2)
    Label(m, text='To: ').grid(row=3)
    start_time_interval_entry = Entry(m)
    start_time_interval_entry.grid(row=2, column=1)
    end_time_interval_entry = Entry(m)
    end_time_interval_entry.grid(row=3, column=1)
    confirmButton = Button(m, text="Confirm", width=16, command=lambda: attempt_update(
        vm,
        ax,
        canvas,
        start_time_interval_entry.get(),
        end_time_interval_entry.get()
    ))
    confirmButton.grid(row=4, column=1)
