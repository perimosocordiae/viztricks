import os
import matplotlib as mpl
from itertools import count
from matplotlib import animation as mpl_animation
from matplotlib import pyplot as plt
from subprocess import check_call

__all__ = ['FigureSaver']


class FigureSaver(object):
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
      self.writer_ctx = self.writer.saving(plt.gcf(), name+'.mp4', dpi)
    else:
      self.fpatt = name + '-%05d.png'
      self.counter = count()
      self.delay = 100 // fps  # delay in hundredths of a second

  def __enter__(self):
    if self.mode == 'video':
      def new_show():
        self.writer.grab_frame()
        plt.clf()
      self.writer_ctx.__enter__()
    else:
      def new_show():
        plt.savefig(self.fpatt % next(self.counter))
        plt.clf()

    # Stash and swap
    self.old_show = plt.show
    plt.show = new_show

  def __exit__(self, *args):
    # restore the old show function
    plt.show = self.old_show
    plt.close()
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
