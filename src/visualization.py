import numpy as np
import matplotlib.pyplot as plt


def plot_fields(fields, labels, max_value, signed):
    fig = plt.figure()
    plt.ylim(-1 * max_value if signed else 0, max_value)
    x = np.arange(fields[0].size)
    plots = [plt.plot(x, f, label=l)[0] for (f, l) in zip(fields, labels)]
    plt.legend(loc='upper right')

    return fig, plots


class LiveViz:
    def __init__(self):
        pass

    def render(self, fields, labels, max_value, signed):
        pass


class LivePlotter(LiveViz):
    def __init__(self):
        super().__init__()
        self.fig = None
        self.plots = None

    def render(self, fields, labels, max_value, signed):
        if self.fig is None:
            plt.ion()
            self.fig, self.plots = plot_fields(fields, labels, max_value, signed)
        else:
            for i in range(len(self.plots)):
                self.plots[i].set_ydata(fields[i])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
