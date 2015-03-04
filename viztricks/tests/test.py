import matplotlib
matplotlib.use('template')  # Mock backend, doesn't show anything

import numpy as np
import unittest
import viztricks as viz
from viztricks import shims


class TestVizTricks(unittest.TestCase):
  # These exercise plotting methods, but don't actually check for correct output
  def setUp(self):
    self.X = np.array([[1,2],[2,1],[3,1.5],[4,0.5],[5,1]])
    self.Y = np.array([[1,2,3],[3,2,1],[3,1.5,2],[4,0.5,-1],[5,1,4]])

  def test_plot(self):
    viz.plot(self.X, '-o', title='Test')
    viz.plot(self.X, scatter=True, c=self.X.sum(axis=1))
    viz.plot(self.X[0])
    viz.plot(self.X[0], scatter=True)
    viz.plot(self.Y, '-o', title='Test')
    viz.plot(self.Y, scatter=True, c=self.Y.sum(axis=1))

  def test_plot_trajectories(self):
    labels = ['a','b','c','d','e']
    viz.plot_trajectories([self.X, self.X+2], '-x', labels=labels, title='Test')
    viz.plot_trajectories([self.Y], '--', labels=labels, title='Test 3D')

  def test_gradient_line(self):
    viz.gradient_line(self.X[:,0], self.X[:,1])

  def test_violinplot(self):
    viz.violinplot(self.Y, showmedians=True)
    shims._violinplot(self.Y, showextrema=True, showmeans=True,
                      showmedians=True)

  def test_vector_field(self):
    viz.vector_field(self.X, -self.X/2, title='arrows')
    viz.vector_field(self.Y, -self.Y/2, title='arrows')

  def test_irregular_contour(self):
    a,b,c = self.Y.T
    viz.irregular_contour(a, b, c)

if __name__ == '__main__':
  unittest.main()
