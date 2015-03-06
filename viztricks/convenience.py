import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'plot', 'plot_trajectories', 'imagesc', 'axes_grid'
]


def plot(X, marker='.', title=None, fig=None, ax=None, scatter=False, **kwargs):
  '''General plotting function for a set of points X.
  May be 1, 2, or 3 dimensional.'''
  assert len(X.shape) in (1,2), 'Only valid for 1 or 2-d arrays of points'
  assert (len(X.shape) == 1 or X.shape[1] in (1,2,3)
          ), 'Only valid for [1-3]-dimensional points'
  is_3d = len(X.shape) == 2 and X.shape[1] == 3
  is_1d = len(X.shape) == 1 or X.shape[1] == 1
  if ax is None:
    if fig is None:
      fig = plt.gcf()
    if is_3d:
      from mpl_toolkits.mplot3d import Axes3D
      ax = Axes3D(fig)
    else:
      ax = fig.add_subplot(111)
  elif is_3d:
    assert hasattr(ax, 'zaxis'), 'Must provide an Axes3D axis'
  # Do the plotting
  if scatter:
    if is_1d:
      ax.scatter(range(len(X)), X, marker=marker, **kwargs)
    elif is_3d:
      ax.scatter(X[:,0], X[:,1], X[:,2], marker=marker, **kwargs)
    else:
      ax.scatter(X[:,0], X[:,1], marker=marker, **kwargs)
  else:
    if is_1d:
      ax.plot(X, marker, **kwargs)
    elif is_3d:
      ax.plot(X[:,0], X[:,1], X[:,2], marker, **kwargs)
    else:
      ax.plot(X[:,0], X[:,1], marker, **kwargs)
  if title:
    ax.set_title(title)
  return plt.show


def plot_trajectories(T, marker='x-', labels=None,
                      title=None, fig=None, ax=None):
  '''Plot t trajectories (2d or 3d).'''
  assert (hasattr(T[0], 'shape') and len(T[0].shape) == 2
          ), 'T must be a sequence of 2d or 3d point-sets'
  plot(T[0], marker=marker, fig=fig, ax=ax)
  # hack: make sure we use the same fig and ax for each plot
  fig = plt.gcf()
  ax = fig.gca()
  for traj in T[1:]:
    plot(traj, marker=marker, fig=fig, ax=ax)
  if labels:
    ax.legend(labels)
  if title:
    ax.set_title(title)
  return plt.show


def imagesc(data, ax=None):
  '''Simple alias for a Matlab-like imshow function.'''
  if ax is None:
    ax = plt.gca()
  ax.imshow(data, interpolation='nearest', aspect='auto')
  return plt.show


def axes_grid(n):
  '''Finds a reasonable arrangement of n axes. Returns (fig, axes) tuple.'''
  r = np.floor(np.sqrt(n))
  r, c = int(r), int(np.ceil(n / r))
  fig, axes = plt.subplots(nrows=r, ncols=c, figsize=(c*4, r*4))
  axes = np.atleast_2d(axes)
  # Turn off any extra axes
  for ax in axes.flat[n:]:
    ax.set_axis_off()
  return fig, axes
