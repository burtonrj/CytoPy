def plot_gate(self, gate_name, xlim=None, ylim=None):
    gate = self.gates[gate_name]
    data = dict(primary=self.get_population_df(gate.parent))
    kwargs = {k: v for k, v in gate.func_args}

    def get_fmo_data(fk):
        if fk in kwargs.keys():
            return self.knn_fmo(gate.parent, kwargs[fk])
        return None

    for x in ['fmo_x', 'fmo_y']:
        d = get_fmo_data(x)
        if d is not None:
            data[x] = d

    n = len(data.keys())
    # Get axis info
    x = gate.x
    if gate.y:
        y = gate.y
    else:
        y = 'FSC-A'
    xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
    if gate.gate_type == 'cluster':
        return self.__cluster_plot(x, y, gate, title=gate.gate_name)
    fig, axes = plt.subplots(ncols=n)
    if n > 1:
        for ax, (name, d) in zip(axes, data.items()):
            self.__geom_plot(ax=ax, x=x, y=y, data=d, geom=self.populations[gate.children[0]]['geom'],
                             xlim=xlim, ylim=ylim, title=f'{gate.gate_name}_{name}')
    else:
        self.__geom_plot(ax=axes, x=x, y=y, data=data['primary'], geom=self.populations[gate.children[0]]['geom'],
                         xlim=xlim, ylim=ylim, title=gate.gate_name)
    return fig, axes


@staticmethod
def __plot_axis_lims(x, y, xlim=None, ylim=None):
    if not xlim:
        if any([x.find(c) != -1 for c in ['FSC', 'SSC']]):
            xlim = (0, 250000)
        else:
            xlim = (0, 1)
    if not ylim:
        if any([y.find(c) != -1 for c in ['FSC', 'SSC']]):
            ylim = (0, 250000)
        else:
            ylim = (0, 1)
    return xlim, ylim


def __cluster_plot(self, x, y, gate, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    colours = cycle(['black', 'green', 'blue', 'red', 'magenta', 'cyan'])
    for child, colour in zip(gate.children, colours):
        d = self.get_population_df(child)
        if d is not None:
            d.sample(frac=0.5)
        else:
            continue
        ax.scatter(d[x], d[y], c=colour, s=3, alpha=0.4)
    ax.set_title(title)
    fig.show()


def __standard_2dhist(self, ax, data, x, y, xlim, ylim, title):
    if data.shape[0] <= 100:
        bins = 50
    elif data.shape[0] > 1000:
        bins = 500
    else:
        bins = int(data.shape[0] * 0.5)
    ax.hist2d(data[x], data[y], bins=bins, norm=LogNorm())
    ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
    return ax


@staticmethod
def __plot_asthetics(ax, x, y, xlim, ylim, title):
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.set_title(title)
    return ax


def __geom_plot(self, ax, x, y, data, geom, xlim, ylim, title):
    if data.shape[0] > 1000:
        ax = self.__standard_2dhist(ax, data, x, y, xlim, ylim, title)
        ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
    else:
        ax.scatter(x=data[x], y=data[y], s=3)
        ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
    if 'threshold' in geom.keys():
        ax.axvline(geom['threshold'], c='r')
    if 'threshold_x' in geom.keys():
        ax.axvline(geom['threshold_x'], c='r')
    if 'threshold_y' in geom.keys():
        ax.axhline(geom['threshold_y'], c='r')
    if all([x in geom.keys() for x in ['mean', 'width', 'height', 'angle']]):
        ellipse = patches.Ellipse(xy=geom['mean'], width=geom['width'], height=geom['height'],
                                  angle=geom['angle'], fill=False, edgecolor='r')
        ax.add_patch(ellipse)
    if all([x in geom.keys() for x in ['x_min', 'x_max', 'y_min', 'y_max']]):
        rect = patches.Rectangle(xy=(geom['x_min'], geom['y_min']),
                                 width=((geom['x_max']) - (geom['x_min'])),
                                 height=(geom['y_max'] - geom['y_min']),
                                 fill=False, edgecolor='r')
        ax.add_patch(rect)
    return ax


def plot_population(self, population_name, x, y, xlim=None, ylim=None, show=True):
    fig, ax = plt.subplots(figsize=(5, 5))
    if population_name in self.populations.keys():
        data = self.get_population_df(population_name)
    else:
        print(f'Invalid population name, must be one of {self.populations.keys()}')
        return None
    xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
    if data.shape[0] < 500:
        ax.scatter(x=data[x], y=data[y], s=3)
        ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
    else:
        self.__standard_2dhist(ax, data, x, y, xlim, ylim, title=population_name)
    if show:
        fig.show()
    return fig