import numpy as np
import matplotlib.pyplot as plt

__all__ = ['violinplot']


def _violin(data, pos, amplitude, **kwargs):
  # http://pyinsci.blogspot.it/2009/09/violin-plot-with-matplotlib.html
  from scipy.stats import gaussian_kde
  # get the value of the parameters
  ax = kwargs.pop('ax', plt.gca())
  # evaluate the violin plot
  x = np.linspace(min(data),max(data),101)  # support for violin
  try:
    v = gaussian_kde(data).evaluate(x)  # violin profile (density curve)
  except:
    v = np.ones_like(x)
  # set the length of the profile
  v /= v.max() / amplitude
  kwargs.setdefault('facecolor','y')
  kwargs.setdefault('alpha',0.33)
  return ax.fill_betweenx(x,pos-v,pos+v,**kwargs)


def _violinplot(dataset, positions=None, vert=True, widths=0.5,
                showmeans=False, showextrema=True, showmedians=False,
                points=100, bw_method=None, ax=None):
  '''Local version of the matplotlib violinplot.'''
  if ax is None:
    ax = plt.gca()
  if positions is None:
    positions = np.arange(1, len(dataset)+1)
  amp = widths / 2.0
  result = dict(bodies=[], means=[], mins=[], maxes=[], bars=[], medians=[])
  for pos, d in zip(positions, dataset):
    result['bodies'].append(_violin(d, pos, amp, ax=ax))
    x0 = pos - amp/2.
    x1 = pos + amp/2.
    d_min, d_max = np.min(d), np.max(d)
    result['bars'].append(ax.vlines(pos, d_min, d_max))
    if showmedians:
      m1 = np.median(d)
      result['medians'].append(ax.plot([x0,x1], [m1,m1], 'k-'))
    if showmeans:
      m1 = np.mean(d)
      result['means'].append(ax.plot([x0,x1], [m1,m1], 'k-'))
    if showextrema:
      result['mins'].append(ax.plot([x0,x1], [d_min,d_min], 'k-'))
      result['maxes'].append(ax.plot([x0,x1], [d_max,d_max], 'k-'))
  ax.set_xticks(positions)
  return result

try:
  from matplotlib.pyplot import violinplot
except ImportError:
  # violinplot was added to master matplotlib in early June 2014.
  violinplot = _violinplot
