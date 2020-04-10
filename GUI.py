from tkinter import *

def setup_gui():
    m = Tk()
    m.title('HeatMap')

    menu_button_frame = Frame()
    menu_buttons_widget(m, menu_button_frame)

    map_frame = Frame()
    heatmap_widget(map_frame)

    widgets_frame = Frame()
    time_widget(widgets_frame)

    menu_button_frame.pack()
    map_frame.pack()
    widgets_frame.pack()

    m.mainloop()


def menu_buttons_widget(parent, m):
    mb = Menubutton(m, text= "File")
    mb.grid()
    mb.menu = Menu(mb, tearoff=0)
    mb["menu"] = mb.menu
    cVar = IntVar()
    aVar = IntVar()
    mb.menu.add_checkbutton(label='Load...', variable=cVar)
    mb.menu.add_checkbutton(label='Exit', command=parent.destroy)
    mb.pack()


def heatmap_widget(m):
    import numpy
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # representation should change but this is a good start for the testing for now :)
    # real data should be brought here and not some random data as seen below.
    x = numpy.random.rand(1000)
    y = numpy.random.rand(1000)

    f = plt.Figure(figsize=(5,5), dpi=100)
    a = f.add_subplot(111)
    a.plot(x, y)
    canvas = FigureCanvasTkAgg(f, m)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)


def time_widget(m):
    Label(m, text="Choose Timeframe (in seconds):").grid(row=1)

    Label(m, text='From ').grid(row=2)
    Label(m, text='To: ').grid(row=3)
    Entry(m).grid(row=2, column=1)
    Entry(m).grid(row=3, column=1)
