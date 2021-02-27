import numpy as np
from data_loader import DataLoader

#import pandas as pd
import time

class LogisticRegressionClassifier :
  def __init__(self, max_iter = 200, lr = 0.001, alpha = 10.0, bs = 32):
    self.max_iter_ =  max_iter
    self.lr_ = lr
    self.alpha_ = alpha
    self.bs_ = bs
    return

  def BGD(self, X, y):
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    X = X.values

    self.log_loss_ = np.array([])
    print("shape ", X.shape, y.shape)
    for it in range(self.max_iter_):
      grad_w = np.zeros(X.shape[1])
      grad_b = 0.0
      log_loss = 0.0
      for i in range (X.shape[0]) :
        if i % 400000 == 0 :
          print ('progress',  1.0 * i / X.shape[0])
        yi = y.iloc[i]
        x = X[i]
        sigma = 1.0 / (1.0 + np.exp(-np.dot(w, x) - b))
        sigma = np.clip(sigma, 1e-8, 1 - 1e-8)
        error = yi- sigma
        grad_w += error * x 
        grad_b += error 
        log_loss -= (yi * np.log(sigma) + (1.0 - yi) * np.log((1.0 - sigma)))

      w += self.lr_ * grad_w - 2.0 * self.alpha_ * w
      b += self.lr_ * grad_b - 2.0 * self.alpha_ * b
      self.log_loss_ = np.append(self.log_loss_, log_loss)
      print ("log_loss of iter {0}:{1}".format(it, log_loss / X.shape[0]))

    self.w_ = w
    self.b_ = b

  def SGD(self, X, y, test_X, test_y):
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    self.w_ = w
    self.b_ = b
    X = X.values
    test_X = test_X.values
    batch_size = 0
    self.log_loss_ = np.array([])
    self.val_log_loss_ = np.array([])
    for it in range(self.max_iter_):
      log_loss = 0.0
      grad_w = np.zeros(X.shape[1])
      grad_b = 0.0
      iter_log_loss = 0.0
      batch_log_loss = 0.0
      batch = 0
      for i in range(X.shape[0]):
        yi = y.iloc[i]
        x = X[i]
        sigma = 1.0 / (1.0 + np.exp(-np.dot(w, x) - b))
        sigma = np.clip(sigma, 1e-8, 1 - 1e-8)
        #sigma_old = 1.0 / (1.0 + np.exp(-np.dot(self.w_, x) - self.b_))
        #sigma_old = np.clip(sigma_old, 1e-8, 1 - 1e-8)
        error = yi- sigma
        grad_w += error * x 
        grad_b += error 
        #iter_log_loss -= yi * np.log(sigma_old) + (1.0 - yi) * np.log((1.0 - sigma_old))
        batch_log_loss -= yi * np.log(sigma) + (1.0 - yi) * np.log((1.0 - sigma))

        batch_size += 1
        if batch_size == self.bs_ :
          batch += 1
          batch_size = 0
          w += self.lr_ * grad_w - 2 * self.alpha_ * w
          b += self.lr_ * grad_b - 2 * self.alpha_ * b
          self.w_ = w
          self.b_ = b
          grad_w = np.zeros(X.shape[1])
          grad_b = 0.0
          
          if batch % 500 == 0:
            print ("iter {0}, batch:{1}, expected train_loss :{2}, val loss : {3},  params: b:{4}".format(it, batch, batch_log_loss / (i + 1), self.log_loss(test_X, test_y), b))

      if batch_size != self.bs_:
        w += self.lr_ * grad_w - 2 * self.alpha_ * w
        b += self.lr_ * grad_b - 2 * self.alpha_ * b

      self.w_ = w
      self.b_ = b

      train_log_loss = self.log_loss(X, y)
      self.log_loss_ = np.append(self.log_loss_, train_log_loss)
      val_log_loss = self.log_loss(test_X, test_y)
      self.val_log_loss_ = np.append(self.val_log_loss_, val_log_loss)

      print ("iter {0}, train log loss:{1}, val log loss:{2}".format(it, train_log_loss, val_log_loss))
    return self

  def SGD_momentum(self, X, y, test_X, test_y, gamma = 0.9):
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    self.w_ = w
    self.b_ = b
    X = X.values
    test_X = test_X.values
    batch_size = 0
    self.log_loss_ = np.array([])
    self.val_log_loss_ = np.array([])

    v = np.zeros(X.shape[1])

    for it in range(self.max_iter_):
      start = time.time()
      log_loss = 0.0
      grad_w = np.zeros(X.shape[1])
      grad_b = 0.0
      iter_log_loss = 0.0
      batch_log_loss = 0.0
      batch = 0
      for i in range(X.shape[0]):
        yi = y.iloc[i]
        x = X[i]
        #sigma = 1.0 / (1.0 + np.exp(-np.dot(w, x) - b))
        #sigma = np.clip(sigma, 1e-8, 1 - 1e-8)
        #sigma_old = 1.0 / (1.0 + np.exp(-np.dot(self.w_, x) - self.b_))
        #sigma_old = np.clip(sigma_old, 1e-8, 1 - 1e-8)
        x_of_sigmoid = np.dot(w, x) + b
        sigma = self.sigmoid(x_of_sigmoid)
        x_of_sigmoid2 = np.dot(self.w_, x) + self.b_
        sigma_old = self.sigmoid(x_of_sigmoid2)

        error = yi- sigma
        grad_w += (0 - error * x + self.alpha_ * self.w_) / self.bs_
        grad_b += (0 - error + self.alpha_ * self.b_) / self.bs_
        #iter_log_loss -= yi * np.log(sigma_old) + (1.0 - yi) * np.log((1.0 - sigma_old))
        #batch_log_loss -= yi * np.log(sigma) + (1.0 - yi) * np.log((1.0 - sigma))
        batch_log_loss += self.cross_entropy(yi, x_of_sigmoid)

        batch_size += 1
        if batch_size == self.bs_ :
          batch += 1
          batch_size = 0

          #w -= self.lr_ * grad_w - 2 * self.alpha_ * w

          v = gamma * v + self.lr_ * grad_w
          w -= v
          b -= self.lr_ * grad_b

          self.w_ = w
          self.b_ = b

          grad_w = np.zeros(X.shape[1])
          grad_b = 0.0
          end = time.time()
          #print('batch exe time:', end - start)
          
          if batch % 500 == 0:
            print ("iter {0}, batch:{1}, expected train_loss :{2}, val loss : {3},  params: b:{4}".format(it, batch, batch_log_loss / (i + 1), self.log_loss(test_X, test_y), b))

          start = time.time()

      if batch_size != self.bs_:

          v = gamma * v + self.lr_ * grad_w
          w -= v
          b -= self.lr_ * grad_b

          self.w_ = w
          self.b_ = b

      self.w_ = w
      self.b_ = b

      train_log_loss = self.log_loss(X, y)
      self.log_loss_ = np.append(self.log_loss_, train_log_loss)
      val_log_loss = self.log_loss(test_X, test_y)
      self.val_log_loss_ = np.append(self.val_log_loss_, val_log_loss)

      print ("iter {0}, train log loss:{1}, val log loss:{2}".format(it, train_log_loss, val_log_loss))
    return self

  def SGD_momentum_batch_matrix(self, X, y, test_X, test_y, gamma = 0.9):
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    self.w_ = w
    self.b_ = b
    X = X.values
    test_X = test_X.values
    batch_size = 0
    self.log_loss_ = np.array([])
    self.val_log_loss_ = np.array([])

    v = np.zeros(X.shape[1])

    for it in range(self.max_iter_):
      log_loss = 0.0
      grad_w = np.zeros(X.shape[1])
      grad_b = 0
      batch_log_loss = 0.0

      data_loader = DataLoader(X, y, self.bs_)
      for i, (X_batch, y_batch)  in enumerate(data_loader):
        logits = np.matmul(X_batch, w) + b
        sigmas = np.array([self.sigmoid(x) for x in logits])
        errors = y_batch - sigmas
        grad_w = -np.matmul(X_batch.T, errors) / y_batch.shape[0] + self.alpha_ * self.w_
        grad_b = -sum(errors) / len(errors) + self.alpha_ * self.b_
        
        #batch_log_loss += sum([self.cross_entropy(y_batch.iloc[j], logits[j]) for j in range(y_batch.shape[0])])

        v = gamma * v + self.lr_ * grad_w
        w -= v
        b -= self.lr_ * grad_b

        self.w_ = w
        self.b_ = b

        if (i+1) % 14000 == 0:
          print ("iter {0}, batch:{1}, expected train_loss :{2}, val loss : {3},  params: b:{4}".format(it, i, batch_log_loss / (self.bs_ * i + 1), self.log_loss(test_X, test_y), b))

      train_log_loss = self.log_loss(X, y)
      self.log_loss_ = np.append(self.log_loss_, train_log_loss)
      val_log_loss = self.log_loss(test_X, test_y)
      self.val_log_loss_ = np.append(self.val_log_loss_, val_log_loss)

      print ("iter {0}, train log loss:{1}, val log loss:{2}".format(it, train_log_loss, val_log_loss))
    return self


  def sigmoid(self, x):
    #return np.clip(1.0 / (1.0 + np.exp(-np.dot(w, x) - b)), 1e-8, 1 - 1e-8)
    if x < 0:
      ex = np.exp(x)
      return ex / (1 + ex)
    else :
      return 1.0 / (1 + np.exp(-x))

  def cross_entropy(self, y, x):
    if x < 0:
      return -y * x + np.log(1.0 + np.exp(x))
    else :
      return (1.0 - y) * x + np.log(1.0 + np.exp(-x))
    #return 0.0 - (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

  def log_loss(self, X, y):
    logloss = 0.0
    #X = X.values()
    for i in range(X.shape[0]):
      x = np.dot(self.w_, X[i]) + self.b_
      #sigma = self.sigmoid(x)
      logloss += self.cross_entropy(y.iloc[i], x)

    return logloss / X.shape[0]

  def fit(self, X, y, val_X, val_Y, method="BGD"):
    if method == "BGD":
      self.BGD(X, y)
    elif method == 'SGD' : 
      self.SGD(X, y, val_X, val_Y)
    elif method == 'SGD_momentum':
      self.SGD_momentum_batch_matrix(X, y, val_X, val_Y)
      #self.SGD_momentum(X, y, val_X, val_Y)
    return self

  def predict(self, X):
    X = X.values
    y = np.array([])
    for i in range (X.shape[0]):
      p = 1.0 / (1.0 + np.exp(-np.dot(self.w_, X[i]) - self.b_))
      if p > 0.5 : 
        y = np.append(y, 1)
      else :
        y = np.append(y, 0)
    return y

  def predict_proba(self, X):
    X = X.values
    y = np.array([])
    for i in range (X.shape[0]):
      p = 1.0 / (1.0 + np.exp(-np.dot(self.w_, X[i]) - self.b_))
      y = np.append(y, p)

    return y

  def dump(self):
    print (self.b_, self.w_)
    return self
