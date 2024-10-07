

def plot_helper(ax, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, legend=False, grid=False):
    if xlim is not None:
        mi, ma = xlim
        ax.set_xlim(mi, ma)
    if ylim is not None:
        mi, ma = ylim
        ax.set_ylim(mi, ma)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend()
    if grid:
        ax.grid(grid)
    return ax
