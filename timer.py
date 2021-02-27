import time

class Timer():
  def __init__(self, name = '', verbose = False):
    self.name_ = name
    self.verbose = verbose
    return

  def __enter__(self):
    self.start_ = time.time()

  def __exit__(self, *args):
    self.end_ = time.time()
    self.secs = self.end_ - self.start_
    if self.verbose :
      print('{0} exe time : {1}ms'.format(self.name_, round(self.secs * 1000.0, 3)))
