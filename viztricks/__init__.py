import os
import numpy as np
import matplotlib as mpl
from itertools import count
from matplotlib import animation as mpl_animation
from matplotlib import pyplot
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon, Ellipse
from subprocess import check_call


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
      fig = pyplot.gcf()
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
  return pyplot.show


def plot_trajectories(T, marker='x-', labels=None,
                      title=None, fig=None, ax=None):
  '''Plot t trajectories (2d or 3d).'''
  assert (hasattr(T[0], 'shape') and len(T[0].shape) == 2
          ), 'T must be a sequence of 2d or 3d point-sets'
  plot(T[0], marker=marker, fig=fig, ax=ax)
  # hack: make sure we use the same fig and ax for each plot
  fig = pyplot.gcf()
  ax = fig.gca()
  for traj in T[1:]:
    plot(traj, marker=marker, fig=fig, ax=ax)
  if labels:
    ax.legend(labels)
  if title:
    ax.set_title(title)
  return pyplot.show


def gradient_line(xs, ys, colormap_name='jet', ax=None):
  '''Plot a 2-d line with a gradient representing ordering.
  See http://stackoverflow.com/q/8500700/10601 for details.'''
  if ax is None:
    ax = pyplot.gca()
  cm = pyplot.get_cmap(colormap_name)
  npts = len(xs)-1
  ax.set_color_cycle([cm(float(i)/npts) for i in xrange(npts)])
  for i in xrange(npts):
    ax.plot(xs[i:i+2],ys[i:i+2])
  return pyplot.show


def half_violin_plot(data, pos, left=False, amplitude=0.33, **kwargs):
  # http://pyinsci.blogspot.it/2009/09/violin-plot-with-matplotlib.html
  from scipy.stats import gaussian_kde
  # get the value of the parameters
  ax = kwargs.pop('ax',pyplot.gca())
  # evaluate the violin plot
  x = np.linspace(min(data),max(data),101)  # support for violin
  try:
    v = gaussian_kde(data).evaluate(x)  # violin profile (density curve)
  except:
    v = np.ones_like(x)
  # set the length of the profile
  v /= v.max() / amplitude * (1 if left else -1)
  kwargs.setdefault('facecolor','y')
  kwargs.setdefault('alpha',0.33)
  return ax.fill_betweenx(x,pos,pos+v,**kwargs)


def _violinplot(dataset, positions=None, vert=True, widths=0.5,
                showmeans=False, showextrema=True, showmedians=False,
                points=100, bw_method=None, ax=None):
  '''Local version of the matplotlib violinplot.'''
  if ax is None:
    ax = pyplot.gca()
  if positions is None:
    positions = np.arange(len(dataset))
  amp = widths / 2.0
  for pos, d1 in zip(positions, dataset):
    half_violin_plot(d1, pos, left=False, ax=ax, amplitude=amp)
    half_violin_plot(d1, pos, left=True, ax=ax, amplitude=amp)
    x0 = pos - amp/2.
    x1 = pos + amp/2.
    if showmedians:
      m1 = np.median(d1)
      ax.plot([x0,x1], [m1,m1], 'k-')
    if showmeans:
      m1 = np.mean(d1)
      ax.plot([x0,x1], [m1,m1], 'k-')
    if showextrema:
      m1,m2 = np.min(d1), np.max(d1)
      ax.plot([x0,x1,None,x0,x1], [m1,m1,None,m2,m2], 'k-')
  ax.set_xticks(positions)
  return pyplot.show

try:
  from matplotlib.pyplot import violinplot
except ImportError:
  # violinplot was added to master matplotlib in early June 2014.
  violinplot = _violinplot


def quiver3d(ax, x, y, z, dx, dy, dz, **kwargs):
  try:
    return ax.quiver(x, y, z, dx, dy, dz, pivot='tail', **kwargs)
  except AttributeError:
    # this mpl doesn't have the pivot kwarg, and it defaults to 'head' behavior
    return ax.quiver(x+dx, y+dy, z+dz, dx, dy, dz, **kwargs)


def vector_field(points, directions, title=None, fig=None, ax=None,
                 edge_style='k-', vertex_style='o'):
  '''Plots vectors that start at 'points', and move along 'directions'.'''
  assert points.shape[1] in (2,3) and directions.shape == points.shape
  # Make sure we have an axis.
  if ax is None:
    if points.shape[1] == 2:
      ax = pyplot.gca()
    else:
      from mpl_toolkits.mplot3d import Axes3D
      if fig is None:
        fig = pyplot.gcf()
      ax = Axes3D(fig)
  # Plot.
  if points.shape[1] == 2:
    x,y = points.T
    dx,dy = directions.T
    if hasattr(ax, 'zaxis'):  # Must be on a 3d plot axis, so supply zeros.
      quiver3d(ax, x, y, 0, dx, dy, 0, arrow_length_ratio=0.1)
    else:
      args = (x, y, dx, dy)
      ax.quiver(*args, angles='xy', scale_units='xy', scale=1, headwidth=5)
    if vertex_style is not None:
      ax.scatter(x, y, marker=vertex_style, zorder=2, edgecolor='none')
  else:
    x,y,z = points.T
    dx,dy,dz = directions.T
    quiver3d(ax, x, y, z, dx, dy, dz, arrow_length_ratio=0.1)
    if vertex_style is not None:
      ax.scatter(x, y, z, marker=vertex_style, zorder=2, edgecolor='none')
  if title:
    ax.set_title(title)
  return pyplot.show


class FigureSaver(object):  # pragma: no cover
  '''Context manager for saving successive figures:
  with FigureSaver(name='sweet-animation', mode='gif'):
    something_that_plots()  # calls pyplot.show
  '''
  def __init__(self, name='plot', mode='frames', fps=2):
    assert mode in ('frames', 'gif', 'video')
    self.mode = mode
    self.name = name
    if mode == 'video':
      for coder_prog in ['avconv', 'ffmpeg']:
        if coder_prog in mpl_animation.writers.avail:
          writer_cls = mpl_animation.writers[coder_prog]
          break
      else:
        raise Exception('No available codec for mode="video"')
      self.writer = writer_cls(fps=fps)
      dpi = mpl.rcParams['savefig.dpi']
      self.writer_ctx = self.writer.saving(pyplot.gcf(), name+'.mp4', dpi)
    else:
      self.fpatt = name + '-%05d.png'
      self.counter = count()
      self.delay = 100 // fps  # delay in hundredths of a second

  def __enter__(self):
    if self.mode == 'video':
      def new_show():
        self.writer.grab_frame()
        pyplot.clf()
      self.writer_ctx.__enter__()
    else:
      def new_show():
        pyplot.savefig(self.fpatt % next(self.counter))
        pyplot.clf()

    # Stash and swap
    self.old_show = pyplot.show
    pyplot.show = new_show

  def __exit__(self, *args):
    # restore the old show function
    pyplot.show = self.old_show
    pyplot.close()
    if self.mode == 'frames':
      return False
    if self.mode == 'video':
      return self.writer_ctx.__exit__(*args)
    # collect the frames for conversion
    total_frames = next(self.counter)
    if total_frames < 2:
      print 'Cannot animate < 2 images. Got %d frames.' % total_frames
      return False
    filenames = [self.fpatt % n for n in xrange(total_frames)]
    # shell out to imagemagick to animate
    check_call(['convert', '-delay', str(self.delay)] + filenames[:-1] +
               ['-delay', '200', filenames[-1], self.name + '.gif'])
    # delete the individual frames
    for f in filenames:
      os.unlink(f)
    return False


def irregular_contour(x, y, z, func=pyplot.contourf, func_kwargs=dict(),
                      grid_size=(100,100), padding_fraction=0.05,
                      interp_method='nearest'):
  '''Handles interpolating irregular data to a grid,
  and plots it using the given func [default: contourf]
  See http://wiki.scipy.org/Cookbook/Matplotlib/Gridding_irregularly_spaced_data
  '''
  from scipy.interpolate import griddata  # Late import; scipy is optional
  x_range = (x.min(), x.max())
  y_range = (y.min(), y.max())
  pad_x = padding_fraction * -np.subtract.reduce(x_range)
  pad_y = padding_fraction * -np.subtract.reduce(y_range)
  grid_x = np.linspace(x_range[0] - pad_x, x_range[1] + pad_x, grid_size[0])
  grid_y = np.linspace(y_range[0] - pad_y, y_range[1] + pad_y, grid_size[1])
  grid_z = griddata((x, y), z, (grid_x[None], grid_y[:,None]),
                    method=interp_method)
  return func(grid_x, grid_y, grid_z, **func_kwargs)


def voronoi_filled(points, colors, show_points=False, padding_fraction=0.05,
                   cmap=None, ax=None, alpha=None, edgecolor=None):
  '''Plots a filled voronoi diagram, using the given points and their colors.'''
  from scipy.spatial import Voronoi  # Late import; scipy is optional
  # Borrowed from http://nbviewer.ipython.org/gist/pv/8037100
  assert points.shape[1] == 2, 'Input points must be 2D'
  vor = Voronoi(points)
  regions = []
  vertices = vor.vertices.tolist()

  center = vor.points.mean(axis=0)
  radius = vor.points.ptp().max()*2

  # Construct a map containing all ridges for a given point
  all_ridges = {}
  for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
    all_ridges.setdefault(p1, []).append((p2, v1, v2))
    all_ridges.setdefault(p2, []).append((p1, v1, v2))

  # Reconstruct infinite regions
  for p1, region in enumerate(vor.point_region):
    verts = vor.regions[region]
    if all(v >= 0 for v in verts):
      # finite region
      regions.append(verts)
      continue

    # reconstruct a non-finite region
    ridges = all_ridges[p1]
    new_region = [v for v in verts if v >= 0]

    for p2, v1, v2 in ridges:
      if v2 < 0:
        v1, v2 = v2, v1
      if v1 >= 0:
        # finite ridge: already in the region
        continue

      # Compute the missing endpoint of an infinite ridge
      t = vor.points[p2] - vor.points[p1]  # tangent
      t /= np.linalg.norm(t)
      n = np.array([-t[1], t[0]])  # normal

      midpoint = vor.points[[p1, p2]].mean(axis=0)
      direction = np.sign((midpoint - center).dot(n)) * n
      far_point = vor.vertices[v2] + direction * radius

      new_region.append(len(vertices))
      vertices.append(far_point.tolist())

    # sort region counterclockwise
    vs = np.asarray([vertices[v] for v in new_region])
    vs -= vs.mean(axis=0)
    angle_order = np.argsort(np.arctan2(vs[:,1], vs[:,0]))
    new_region = np.array(new_region)[angle_order]

    # finish
    regions.append(new_region)
  vertices = np.asarray(vertices)

  # Plot colored polygons
  if ax is None:
    ax = pyplot.gca()
  polys = PatchCollection([Polygon(vertices[region]) for region in regions],
                          cmap=cmap, alpha=alpha, edgecolor=edgecolor)
  polys.set_array(colors)
  ax.add_collection(polys)

  if show_points:
    ax.plot(points[:,0], points[:,1], 'ko')

  # Zoom to a reasonable scale.
  pad = padding_fraction * (vor.max_bound - vor.min_bound)
  mins = vor.min_bound - pad
  maxes = vor.max_bound + pad
  ax.set_xlim(mins[0], maxes[0])
  ax.set_ylim(mins[1], maxes[1])
  return polys


def pca_ellipse(data, loc=None, ax=None, **ellipse_kwargs):
  '''Finds the 2d PCA ellipse of given data and plots it.
  loc: center of the ellipse [default: mean of the data]
  '''
  from sklearn.decomposition import PCA  # Late import; sklearn is optional
  pca = PCA(n_components=2).fit(data)
  if loc is None:
    loc = pca.mean_
  if ax is None:
    ax = pyplot.gca()
  cov = pca.explained_variance_ * pca.components_.T
  u,s,v = np.linalg.svd(cov)
  width,height = 2*np.sqrt(s[:2])
  angle = np.rad2deg(np.arctan2(u[1,0], u[0,0]))
  ell = Ellipse(xy=loc, width=width, height=height, angle=angle,
                **ellipse_kwargs)
  ax.add_patch(ell)
  return ell


def imagesc(data, ax=None):
  '''Simple alias for a Matlab-like imshow function.'''
  if ax is None:
    ax = pyplot.gca()
  ax.imshow(data, interpolation='nearest', aspect='auto')
  return pyplot.show


def axes_grid(n):
  '''Finds a reasonable arrangement of n axes. Returns (fig, axes) tuple.'''
  r = np.floor(np.sqrt(n))
  r, c = int(r), int(np.ceil(n / r))
  fig, axes = pyplot.subplots(nrows=r, ncols=c, figsize=(c*4, r*4))
  # Turn off any extra axes
  for ax in axes.flat[n:]:
    ax.set_axis_off()
  return fig, np.atleast_2d(axes)


def embedded_images(X, images, exclusion_radius=None, ax=None, cmap=None,
                    zoom=1, seed=None, frameon=False):
  '''Plots a subset of images on an axis. Useful for visualizing image
  embeddings, especially when plotted over a scatterplot. Selects random points
  to annotate with their corresponding image, respecting an exclusion_radius
  around each selected point.'''
  from sklearn.metrics import pairwise_distances  # TODO: remove this dep.
  assert X.shape[0] == images.shape[0], 'Unequal number of points and images'
  if ax is None:
    ax = pyplot.gca()
  if exclusion_radius is None:
    # TODO: make a smarter default based on image size and axis limits
    exclusion_radius = 1.
  if seed is not None:
    np.random.seed(seed)
  while X.shape[0] > 0:
    i = np.random.choice(X.shape[0])
    im = OffsetImage(images[i], zoom=zoom, cmap=cmap)
    ab = AnnotationBbox(im, X[i], xycoords='data', frameon=frameon)
    ax.add_artist(ab)
    mask = (pairwise_distances(X[i:i+1], X) > exclusion_radius).ravel()
    X = X[mask]
    images = images[mask]
  return pyplot.show
