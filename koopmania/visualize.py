import numpy as np
import koopmania.system as ksys


class SystemViewer:
    default_settings = {'xlim': (-1.0, 1.0),
                        'ylim': (-1.0, 1.0),
                        'visible_dims': (0, 1),
                        'resolution': 10}

    def __init__(self,
                 system: ksys.ContinuousSystem,
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
        self.system: ksys.ContinuousSystem = system

    def get_settings(self, settings):
        return {**self.default_settings, **self.init_settings, **settings}

    def get_state(self, y, settings):
        x = np.zeros(self.system.dimension)
        for idx, v in settings.get('fixed_dims', {}):
            x[idx] = v
        for idx, v in zip(settings['visible_dims'], y):
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


def arrow_plot(ax, arrow_fcn, xlim, ylim, number, **kwargs):
    xmin, xmax = xlim
    ymin, ymax = ylim
    x,y = np.meshgrid(np.linspace(xmin, xmax, number),np.linspace(ymin,ymax,number))
    yall = np.vstack((x.ravel(), y.ravel())).T
    arrows = (np.array([arrow_fcn(y) for y in yall])).reshape((*x.shape, 2))
    u = arrows[:, :, 0]
    v = arrows[:, :, 1]
    defaults = {'alpha': 0.2}
    defaults.update(kwargs)
    ax.quiver(x, y, u, v, **defaults)


def training_data_plot(ax, X, Xp, scatter_kwargs=None, arrow_kwargs=None):
    scatter = {'s': 5, 'label': 'Training Data'}
    if scatter_kwargs is not None:
        scatter.update(scatter_kwargs)

    arrow = {'width': 0.009, 'head_width': 0.04, 'length_includes_head':True, 'edgecolor':'b'}
    if arrow_kwargs is not None:
        scatter.update(arrow_kwargs)

    # plot training data
    ax[0].scatter(*np.hstack((X, Xp)), **scatter)

    for x0, x1 in zip(X.T, Xp.T):
      ax[0].arrow(*x0, *(x1-x0), **arrow)