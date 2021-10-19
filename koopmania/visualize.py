import numpy as np


class SystemViewer:
    default_settings = {'xlim': (-1.0, 1.0),
                        'ylim': (-1.0, 1.0),
                        'visible_dims': (0, 1),
                        'resolution': 10}

    def __init__(self,
                 system: 'ContinuousSystem',
                 xlim= None,
                 ylim=None,
                 visible_dims=None,
                 fixed_dims=None,
                 resolution=None):

        self.init_settings = {'xlim': xlim,
                              'ylim': ylim,
                              'visible_dims': visible_dims,
                              'fixed_dims': fixed_dims,
                              'resolution': resolution}
        self.init_settings = {k: v for k, v in self.init_settings.items() if v is not None}
        self.system: 'ContinuousSystem' = system

    def get_settings(self, settings):
        return {**self.default_settings, **self.init_settings, **settings}

    def get_state(self, y, settings):
        x = np.zeros(self.system.dimension)
        for idx, v in settings.get('fixed_dims', {}):
            x[idx] = v
        for idx, v in zip(settings.get('visible_dims'), y):
            x[idx] = v
        return x

    def plot_quiver(self, ax, quiver_args = None, **settings):
        fsetting = self.get_settings(settings)

        def arrow_fcn(y):
            x = self.get_state(y, fsetting)
            idxs = list(fsetting['visible_dims'])
            return self.system.gradient(0.0, x)[idxs]

        xlim = fsetting['xlim']
        ylim = fsetting['ylim']
        res = fsetting['resolution']

        arrow_plot(ax, arrow_fcn, xlim, ylim, res, **(quiver_args if quiver_args is not None else {}))
        self.apply_ax(ax)

    def apply_ax(self, ax, **settings):
        fsetting = self.get_settings(settings)
        xlim = fsetting['xlim']
        ylim = fsetting['ylim']
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def plot_training(self, ax, Xt, Xtp, scatter_kwargs = None, arrow_kwargs = None, **settings):
        fsetting = self.get_settings(settings)
        akw = {} if arrow_kwargs is None else arrow_kwargs
        skw = {} if scatter_kwargs is None else scatter_kwargs
        visible_dims = fsetting['visible_dims']
        training_data_plot(ax, Xt, Xtp, visible_dims, scatter_kwargs=skw, arrow_kwargs=akw)
        self.apply_ax(ax)

    def plot_trajectory(self, ax, traj, plot_kwargs=None, **settings):
        fsetting = self.get_settings(settings)
        plot = {'label': 'trajectory'}
        if plot_kwargs is not None:
            plot.update(plot_kwargs)

        visible_dims = fsetting['visible_dims']
        print(visible_dims)
        tr = traj[list(visible_dims), :]
        ax.plot(*tr, **plot)


def arrow_plot(ax, arrow_fcn, xlim, ylim, number, **kwargs):
    xmin, xmax = xlim
    ymin, ymax = ylim
    x,y = np.meshgrid(np.linspace(xmin, xmax, number),np.linspace(ymin,ymax,number))
    yall = np.vstack((x.ravel(), y.ravel())).T
    arrows = (np.array([arrow_fcn(y) for y in yall])).reshape((*x.shape, 2))
    u = arrows[:, :, 0]
    v = arrows[:, :, 1]
    defaults = {'alpha': 0.2, 'width': 0.005}
    defaults.update(kwargs)
    ax.quiver(x, y, u, v, **defaults)


def training_data_plot(ax, X, Xp, visible_dims, scatter_kwargs=None, arrow_kwargs=None):
    scatter = {'s': 5, 'label': 'Training Data'}
    X, Xp = X[list(visible_dims), :], Xp[list(visible_dims), :]

    if scatter_kwargs is not None:
        scatter.update(scatter_kwargs)

    arrow = {'width': 0.009, 'head_width': 0.04, 'length_includes_head':True, 'edgecolor':'b', 'alpha': 0.5}
    if arrow_kwargs is not None:
        arrow.update(arrow_kwargs)

    # plot training data
    ax.scatter(*np.hstack((X, Xp)), **scatter)

    for x0, x1 in zip(X.T, Xp.T):
      ax.arrow(*x0, *(x1-x0), **arrow)