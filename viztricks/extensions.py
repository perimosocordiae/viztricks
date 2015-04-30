import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon, Ellipse

__all__ = [
    'gradient_line', 'irregular_contour',
    'voronoi_filled', 'pca_ellipse', 'embedded_images'
]


def gradient_line(xs, ys, colormap_name='jet', ax=None):
  '''Plot a 2-d line with a gradient representing ordering.
  See http://stackoverflow.com/q/8500700/10601 for details.'''
  if ax is None:
    ax = plt.gca()
  cm = plt.get_cmap(colormap_name)
  npts = len(xs)-1
  ax.set_color_cycle([cm(float(i)/npts) for i in xrange(npts)])
  for i in xrange(npts):
    ax.plot(xs[i:i+2],ys[i:i+2])
  return plt.show


def irregular_contour(x, y, z, func=plt.contourf, func_kwargs=dict(),
                      grid_size=(100,100), padding_fraction=0.05,
                      interp_method='nearest'):
  '''Handles interpolating irregular data to a grid,
  and plots it using the given func [default: contourf]
  See http://wiki.scipy.org/Cookbook/Matplotlib/Gridding_irregularly_spaced_data
  '''
  from scipy.interpolate import griddata  # Late import; scipy is optional
  x, y, z = map(np.asanyarray, (x, y, z))
  x_range = (x.min(), x.max())
  y_range = (y.min(), y.max())
  pad_x = padding_fraction * -np.subtract.reduce(x_range)
  pad_y = padding_fraction * -np.subtract.reduce(y_range)
  grid_x = np.linspace(x_range[0] - pad_x, x_range[1] + pad_x, grid_size[0])
  grid_y = np.linspace(y_range[0] - pad_y, y_range[1] + pad_y, grid_size[1])
  grid_z = griddata((x, y), z, (grid_x[None], grid_y[:,None]),
                    method=interp_method)
  return func(grid_x, grid_y, grid_z, **func_kwargs)


def voronoi_filled(points_or_voronoi, colors, show_points=False,
                   padding_fraction=0.05, cmap=None, ax=None, alpha=None,
                   edgecolor=None):
  '''Plots a filled voronoi diagram, using the given points and their colors.
  The first parameter must be an array-like or a scipy.stats.Voronoi object.
  '''
  from scipy.spatial import Voronoi  # Late import; scipy is optional

  # Disambiguate the first parameter
  if isinstance(points_or_voronoi, Voronoi):
    vor = points_or_voronoi
  else:
    points = np.asanyarray(points_or_voronoi)
    assert points.shape[1] == 2, 'Input points must be 2D'
    vor = Voronoi(points)

  # Borrowed from http://nbviewer.ipython.org/gist/pv/8037100
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
    ax = plt.gca()
  polys = PatchCollection([Polygon(vertices[region]) for region in regions],
                          cmap=cmap, alpha=alpha, edgecolor=edgecolor)
  polys.set_array(np.asanyarray(colors))
  ax.add_collection(polys)

  if show_points:
    ax.plot(vor.points[:,0], vor.points[:,1], 'ko')

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
    ax = plt.gca()
  cov = pca.explained_variance_ * pca.components_.T
  u,s,v = np.linalg.svd(cov)
  width,height = 2*np.sqrt(s[:2])
  angle = np.rad2deg(np.arctan2(u[1,0], u[0,0]))
  ell = Ellipse(xy=loc, width=width, height=height, angle=angle,
                **ellipse_kwargs)
  ax.add_patch(ell)
  return ell


def embedded_images(X, images, exclusion_radius=None, ax=None, cmap=None,
                    zoom=1, seed=None, frameon=False):
  '''Plots a subset of images on an axis. Useful for visualizing image
  embeddings, especially when plotted over a scatterplot. Selects random points
  to annotate with their corresponding image, respecting an exclusion_radius
  around each selected point.'''
  assert X.shape[0] == images.shape[0], 'Unequal number of points and images'
  assert X.shape[1] == 2, 'X must be 2d'
  if ax is None:
    ax = plt.gca()
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
    dist = np.sqrt(np.square(X[i] - X).sum(axis=1))
    mask = (dist > exclusion_radius).ravel()
    X = X[mask]
    images = images[mask]
  return plt.show
