class DataLoader():
  def __init__(self, X, y, batch_size):
    self.X_ = X
    self.y_ = y
    self.bs_ = batch_size
    self.batch_index_ = 0
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):
    start = self.i * self.bs_
    if start >= self.X_.shape[0] :
      self.i = 0
      raise StopIteration

    end = min(start + self.bs_, self.X_.shape[0])
    self.i += 1
    #return (self.X_[start : end], self.y_.iloc[start : end])
    return (self.X_[start : end], self.y_[start : end])
