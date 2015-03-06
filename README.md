# viztricks
Recipes and helper functions for plotting with Python + matplotlib


## Installation

To install, clone this repository and run:

    python setup.py install

This package depends on numpy, scipy, and matplotlib.
The `pca_ellipse` function relies on scikit-learn as well, but this is an
optional dependency.
To create animated GIFs with `FigureSaver`, imagemagick must be installed
with the `convert` program on the path.

To run the test suite:

    ./run_tests.sh

Testing requires python packages nose and nose-cov.

## Tricks

viztricks is a collection of helper functions which break down into
some rough categories:

**Convenience**

 * `plot(X, ...)`: handles most of the common cases when plotting a
   set of `n` points represented by a `(n,d)`-shaped numpy array.
 * `plot_trajectories(T, ...)`: similar to `plot`, but takes a list of
   numpy arrays to plot as disjoint trajectories.
 * `imagesc(X, ...)`: recreates the behavior of Matlab's `imagesc` function.
 * `axes_grid(n)`: constructs a grid of `n` subplots,
   laid out for space efficiency.
 * `vector_field(pts, dirs, ...)`: handles most of the common cases when
   working with `quiver` or `quiver3d`.

**Shims**

 * `violinplot`: supplies violin plots,
   which were added in matplotlib's 1.4 release.

**Extensions**

 * `gradient_line(x, y, ...)`: plots a simple 2d line, but colors each
   segment according to a colormap based on point order.
 * `irregular_contour(x, y, z, ...)`: mimics `plt.contourf`, but allows the
   `(x,y)` data points to be irregularly spaced.
 * `voronoi_filled(pts, colors)`: computes a Voronoi diagram, with each point's
   voronoi cell colored.
 * `pca_ellipse(X, ...)`: computes and plots an ellipse representing the first
   two principal components of X. Useful for overlaying on scatter plots.
 * `embedded_images(X, images, ...)`: selects a non-overlapping subset of images
   at locations X and displays them in the figure.

**Utilities**

  * `FigureSaver` is a context manager for converting code that plots and shows
    multiple figures into a no-frills animation creator.
