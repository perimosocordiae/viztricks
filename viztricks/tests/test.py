import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('template')  # Mock backend, doesn't show anything
import unittest
import viztricks as viz
from viztricks import shims

try:
  import sklearn
except ImportError:
  has_sklearn = False
else:
  has_sklearn = True

try:
  import scipy
except ImportError:
  has_scipy = False
else:
  has_scipy = True


class TestVizTricks(unittest.TestCase):
  # These exercise plotting methods, but don't actually check for correct output
  def setUp(self):
    self.X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    self.Y = np.array([[1,2,3],[3,2,1],[3,1.5,2],[4,0.5,-1],[5,1,4]])

  def test_plot(self):
    viz.plot(self.X, '-o', title='Test')
    viz.plot(self.X, scatter=True, c=self.X.sum(axis=1))
    viz.plot(self.X[0])
    viz.plot(self.X[0], kind='scatter')
    viz.plot(self.Y, '-o', title='Test')
    viz.plot(self.Y, scatter=True, c=self.Y.sum(axis=1))
    viz.plot(self.Y, fig='new')
    viz.plot(self.Y, fig=plt.gcf())
    self.assertRaises(ValueError, viz.plot, self.Y, kind='foobar')
    self.assertRaises(ValueError, viz.plot, np.zeros((1,2,3)))

  def test_plot_trajectories(self):
    viz.plot_trajectories([self.X, self.X+2], colors=[1, 2], colorbar=True)
    viz.plot_trajectories([], title='test')

  def test_imagesc(self):
    viz.imagesc(self.X)
    viz.imagesc(self.X, ax=plt.gca(), title='test')

  def test_axes_grid(self):
    fig, axes = viz.axes_grid(1)
    self.assertEqual(axes.shape, (1,1))
    fig, axes = viz.axes_grid(5)
    self.assertEqual(axes.shape, (2,3))
    self.assertTrue(axes[0,0].axison)
    self.assertFalse(axes[-1,-1].axison)

  def test_gradient_line(self):
    viz.gradient_line(self.X[:,0], self.X[:,1])

  def test_violinplot(self):
    viz.violinplot(self.Y, showmedians=True)
    shims._violinplot(self.Y, showextrema=True, showmeans=True,
                      showmedians=True)

  def test_vector_field(self):
    viz.vector_field(self.X, -self.X/2, title='arrows')
    viz.vector_field(self.Y, -self.Y/2, title='arrows')
    # test 2d plot on a 3d axis
    ax = plt.subplot(111, projection='3d')
    viz.vector_field(self.X, -self.X/2, ax=ax)

  def test_irregular_contour(self):
    a,b,c = self.Y.T
    viz.irregular_contour(a, b, c)
    # Test non-ndarray inputs as well.
    viz.irregular_contour(a, b, range(len(c)))

  @unittest.skipUnless(has_scipy, 'requires scipy')
  def test_voronoi_filled(self):
    colors = np.arange(len(self.X))
    viz.voronoi_filled(self.X, colors, show_points=True)
    vor = scipy.spatial.Voronoi(self.X)
    viz.voronoi_filled(vor, colors, ax=plt.gca())

  @unittest.skipUnless(has_sklearn, 'requires scikit-learn')
  def test_pca_ellipse(self):
    ell = viz.pca_ellipse(self.X)
    self.assertAlmostEqual(ell.angle, 165.0567, places=4)
    self.assertAlmostEqual(ell.width, 2.9213, places=4)
    self.assertAlmostEqual(ell.height, 0.7115, places=4)
    viz.pca_ellipse(self.Y, loc=(0,0), ax=plt.gca())

  def test_embedded_images(self):
    images = np.random.random((len(self.X), 3, 3))
    viz.embedded_images(self.X, images, seed=1234)

  def test_FigureSaver(self):
    old_savefig = plt.savefig
    saved_fnames = []
    plt.savefig = lambda fname: saved_fnames.append(fname)
    with viz.FigureSaver(name='test', mode='frames'):
      plt.show()
      plt.show()
    plt.savefig = old_savefig
    self.assertEqual(saved_fnames, ['test-00000.png', 'test-00001.png'])

  def test_jitterplot(self):
    data = [np.arange(4), [1,1,1,2,3,1], [3,4]]
    viz.jitterplot(data, positions=[4,5,6], alpha=0.5)
    viz.jitterplot(data, vert=False, scale=0)

if __name__ == '__main__':
  unittest.main()
