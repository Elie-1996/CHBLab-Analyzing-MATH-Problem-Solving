from tkinter import *
from Visualization import VisualizationMap


def setup_gui(vm: VisualizationMap):
    m = Tk()
    m.title('HeatMap')

    menu_button_frame = Frame()
    menu_buttons_widget(m, menu_button_frame)

    map_frame = Frame()
    heatmap_widget(map_frame, vm)

    widgets_frame = Frame()
    time_widget(widgets_frame)

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


def heatmap_widget(m, vm):
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # representation should change but this is a good start for the testing for now :)
    # real data should be brought here and not some random data as seen below.

    x_coords = []
    y_coords = []
    for current_bin in vm.bins:
        for (x, y) in current_bin:
            x_coords.append(x)
            y_coords.append(y)
    f = plt.Figure(figsize=(5, 5), dpi=100)
    a = f.add_subplot(111)
    a.imshow(vm.full_image)
    a.plot(x_coords, y_coords)
    canvas = FigureCanvasTkAgg(f, m)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


def time_widget(m):
    Label(m, text="Choose Timeframe (in seconds):").grid(row=1)

    Label(m, text='From ').grid(row=2)
    Label(m, text='To: ').grid(row=3)
    Entry(m).grid(row=2, column=1)
    Entry(m).grid(row=3, column=1)
