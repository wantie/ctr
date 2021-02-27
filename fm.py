import numpy as np
#import cupy as cp
from data_loader import DataLoader

#import pandas as pd
import time
from timer import Timer
from functools import reduce
import math

from multiprocessing.dummy import Pool as ThreadPool
from sklearn.metrics import auc, roc_curve

class FactorialMachineClassifier:
  def __init__(self, max_iter = 200, lr = 0.001, lambda_1 = 10.0, lambda_2 = 0.0, bs = 32, k = 50, gamma = 0.9):
    self.max_iter_ =  max_iter
    self.lr0_ = lr
    self.lr_ = lr
    self.lambda_1_ = lambda_1
    self.lambda_2_ = lambda_2
    self.bs_ = bs
    self.gamma_ = gamma
    self.k_ = k

    #self.pool_ = ThreadPool(10)
    return

  def get_learn_rate(self, epoch):
    step = 3
    l = int(epoch / step)

    return math.pow(0.75, l) * self.lr0_


  def SGD_momentum(self, X, y, test_X, test_y):
    self.w_ = np.random.normal(0, 0.01, X.shape[1])
    self.b_ = np.random.normal(0, 0.01, 1)
    self.V_ = np.random.normal(0, 0.01, (X.shape[1], self.k_))

    X = X.values
    y = y.values
    test_X = np.array(test_X.values)
    test_y = np.array(test_y)
    self.log_loss_ = np.array([])
    self.val_log_loss_ = np.array([])

    v = np.zeros(X.shape[1])
    v_V = np.zeros((X.shape[1], self.k_))

    for it in range(self.max_iter_):
      log_loss = 0.0
      grad_V = np.zeros((X.shape[1], self.k_))
      grad_w = np.zeros(X.shape[1])
      grad_b = 0
      batch_log_loss = 0.0
      #self.lr_ = math.pow(0.80, it) * self.lr0_
      self.lr_ = self.get_learn_rate(it)

      v_last = v.copy()
      v_V_last = v_V.copy()
      w_last = self.w_.copy()
      b_last = self.b_.copy()
      V_last = self.V_.copy()

      data_loader = DataLoader(X, y, self.bs_)
      XV = np.zeros((self.bs_, self.k_))

      for bi, (X_batch, y_batch)  in enumerate(data_loader):
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
	
	
        with Timer('batch', False) as t1:
          XV = np.matmul(X_batch, self.V_)
          logits = np.matmul(X_batch, self.w_) + self.b_
          

          
          #for l in range(logits.shape[0]):
          #  #Vx = (X_batch[l]*self.V_.T).T
          #  #logits[l] += 0.5 * (np.square(Vx.sum()) - np.square(Vx).sum())
          #  #logits[l] += 0.5 * (np.square(Vx) - np.sum().sum()
          #  square_of_sum = np.square(np.dot(X_batch[l],self.V_))
          #  sum_of_square = np.dot(np.square(X_batch[l]), np.square(self.V_))

          #  #sum_of_square = np.sum((np.square(X_batch[l]) * np.square(self.V_.T)).T)
          #  #sum_of_square = np.sum((np.square(X_batch[l]) * np.square(self.V_.T)).T)
          #  logits[l] += 0.5 * np.sum(square_of_sum - sum_of_square)
          square_of_sum = np.square(np.dot(X_batch, self.V_))
          sum_of_square = np.dot(np.square(X_batch), np.square(self.V_))
          logits += 0.5 * np.sum(square_of_sum - sum_of_square, axis = 1)

          #f_vx = lambda x,V : (x*V.T).T
          #vf_vx = np.vectorize(f_vx, excluded=[1], signature = '(m)->(a,b)')
          #with Timer('einsum rewrite', True):
          #  #Vxs = np.einsum('ij,jk->ijk', X_batch, self.V_)
          #  Vxs = vf_vx(X_batch, self.V_)
          #  #for l in range(logits.shape[0]):
          #  #  logits[l] += 0.5 * (np.square(Vxs[l].sum()) - np.square(Vxs[l]).sum())

          sigmas = np.array([self.sigmoid(x) for x in logits])
          errors = y_batch - sigmas
          grad_w = -np.matmul(X_batch.T, errors)/y_batch.shape[0]+self.lambda_1_*self.w_
          grad_b = -sum(errors) / len(errors) + self.lambda_1_ * self.b_
         
          #grad_V_func = lambda l : -errors[l] * (np.outer(X_batch[l], XV[l]) - np.matmul(np.diag(X_batch[l]**2), self.V_)) 
          grad_V_func = lambda l : -errors[l] * (np.outer(X_batch[l], XV[l]) - (X_batch[l]**2 * self.V_.T).T)
          grad_V = sum(map(grad_V_func, range(logits.shape[0]))) + self.lambda_2_ * self.V_
          #v = self.gamma_ * v + self.lr_ * grad_w
          #v_V = self.gamma_ * v_V + self.lr_ * (grad_V / y_batch.shape[0])
          v = self.gamma_ * v + (1.0 - self.gamma_) * grad_w
          v_V = self.gamma_ * v_V + (1.0 - self.gamma_ / y_batch.shape[0]) * grad_V

          self.w_ -= self.lr_ * v
          self.b_ -= self.lr_ * grad_b
          self.V_ -= self.lr_ * v_V
         
          batch_log_loss += sum([self.cross_entropy(y_batch[j], logits[j]) for j in range(y_batch.shape[0])])

          if bi % 25000 == 0:
            test_log_loss = 0.0
            #test_log_loss = self.log_loss(test_X, test_y)
            print ("iter {0}, batch:{1}, expected train_loss :{2}, val loss : {3}, params: b:{4}, lr:{5}".format(it, bi, batch_log_loss / (self.bs_ * (bi + 1)), test_log_loss, self.V_[0], self.lr_))

      train_log_loss,train_auc = self.log_loss(X, y, with_auc = True)
      #train_log_loss = 0.0
      self.log_loss_ = np.append(self.log_loss_, train_log_loss)
      val_log_loss,auc = self.log_loss(test_X, test_y, with_auc  = True)
      self.val_log_loss_ = np.append(self.val_log_loss_, val_log_loss)
      print ("iter {0}, train log loss:{1}, val log loss:{2}, train_auc:{3}, auc:{4}".format(it, train_log_loss, val_log_loss, train_auc, auc))

      #retry = input('iterate rerun: [y/n]?')
      #if retry == 'y' :
      #  v = v_last
      #  v_V = v_V_last
      #  self.b_ = b_last
      #  self.w_ = w_last
      #  self.V_ = V_last
      #  self.lr_ = float(input('learn rate:'))

    return self
  
  def sigmoid(self, x):
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

  def logit(self, x):
    ret = 0
    return ret

  def log_loss(self, X, y, with_auc = False):
    logloss = 0.0

    with Timer('log loss', False) :
      #X = X.values()
      with Timer('logloss linear matmul'):
        logits = np.matmul(X, self.w_) + self.b_

      #for l in range (X.shape[0]):
      #  #Vx = np.matmul(np.diag(X[l]), self.V_)
      #  #Vx = (X[l] * self.V_.T).T
      #  #logits[l] += 0.5 * (np.square(Vx.sum()) - np.square(Vx).sum())
      #  square_of_sum = np.square(np.dot(X[l], self.V_))
      #  sum_of_square = np.dot(np.square(X[l]), np.square(self.V_))
      #  logits[l] += 0.5 * np.sum(square_of_sum - sum_of_square)
      #  logloss += self.cross_entropy(y[l], logits[l])

      square_of_sum = np.square(np.dot(X, self.V_))
      sum_of_square = np.dot(np.square(X), np.square(self.V_))
      logits += 0.5 * np.sum(square_of_sum - sum_of_square, axis = 1)
      for l in range(X.shape[0]):
        logloss += self.cross_entropy(y[l], logits[l])

    if with_auc :
      predict_y = np.array([self.sigmoid(l) for l in logits])
      fpr, tpr, thresholds = roc_curve(y, predict_y)
      return logloss / X.shape[0], auc(fpr, tpr)

    return logloss / X.shape[0]

  def fit(self, X, y, val_X, val_Y):
    self.SGD_momentum(X, y, val_X, val_Y)
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

    logits = np.matmul(X, self.w_) + self.b_

    #for l in range (X.shape[0]):
    #  #Vx = np.matmul(np.diag(X[l]), self.V_)
    #  #Vx = (X[l] * self.V_.T).T
    #  square_of_sum = np.square(np.dot(X[l], self.V_))
    #  sum_of_square = np.dot(np.square(X[l]), np.square(self.V_))
    #  logits[l] += 0.5 * np.sum(square_of_sum - sum_of_square)

    square_of_sum = np.square(np.dot(X, self.V_))
    sum_of_square = np.dot(np.square(X), np.square(self.V_))
    logits += 0.5 * np.sum(square_of_sum - sum_of_square, axis = 1)

    
    print(logits.shape)
    y = np.array([self.sigmoid(l) for l in logits])
    print
    return y

  def dump(self):
    print (self.b_, self.w_, self.V_)
    return self
