import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from callbacks import RocCallback

#pd.options.display.max_rows = None

class FmLayer(keras.layers.Layer):
  def __init__(self, k=4, stddev = 0.01, lambda_1 = 0.0000, lambda_2 = 0.00001):
    super(FmLayer, self).__init__()
    self.k = k
    self.stddev = stddev
    self.lambda_1 = lambda_1
    self.lambda_2 = lambda_2

  def build(self, input_shape):
    print('im in build')
    print(input_shape)
    self.b = tf.Variable(tf.random.normal(shape = (1,), stddev = self.stddev), trainable = True)
    self.w = tf.Variable(tf.random.normal(shape = (input_shape[-1],1), stddev = self.stddev), trainable = True)
    self.V = tf.Variable(tf.random.normal(shape = (input_shape[-1], self.k), stddev = self.stddev), trainable = True)
    print(self.V.shape)

  def call(self, inputs):
    print(inputs)
    linear = tf.matmul(inputs, self.w) + self.b

    square_of_sum = tf.square(tf.matmul(inputs, self.V))
    sum_of_square = tf.matmul(tf.square(inputs), tf.square(self.V))

    cross = tf.reduce_sum(square_of_sum - sum_of_square, axis = 1) * 0.5
    cross_2 = tf.expand_dims(cross, -1)
    logits = linear + cross_2
    #print('linear:',linear.shape)
    #print('cross:', cross.shape)
    #print('logits:', logits.shape)

    l2_loss_w = self.lambda_1 * tf.reduce_sum(tf.square(self.w))
    l2_loss_V = self.lambda_2 * tf.reduce_sum(tf.square(self.V))
    self.add_loss(l2_loss_w)
    self.add_loss(l2_loss_V)

    return tf.sigmoid(logits)


def get_data():
  data = pd.read_csv("./data/train.csv.bak")#.head(10000)
  print('columns :', data.columns)

  drop_columns = [
    'id', 'device_id','device_ip','device_model',
    'site_id', 'site_domain',
    'app_id', 'app_domain'
  ]

  print('shape before drop : ', data.shape)
  data.drop(drop_columns, axis=1, inplace = True)
  print('shape after drop : ', data.shape)

  #for col in data.columns:
  #  print("value count of {0}\n{1}".format(col, data[col].value_counts()))


  dummy_dfs = []

  for col in data.columns :
    if col != 'click':
      dummy_dfs.append(pd.get_dummies(data[col], prefix=col))

  X = pd.concat(dummy_dfs, axis = 1)
  print('shape of dummy train_set :', X.shape)
  y = data['click']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=3)

  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_data()
print(X_train.dtypes)

print('a')
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
print('b')
#dataset = dataset.shuffle(800000, reshuffle_each_iteration = True)
print('c')
dataset = dataset.batch(32)
print('d')

valset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10000)

inputs = keras.Input(shape = (X_train.shape[1],))
outputs = FmLayer(32)(inputs)
model = keras.Model(inputs,outputs)
model.summary()

#model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
opt = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
#opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = opt, 
              loss = 'binary_crossentropy',
              metrics = [tf.keras.metrics.AUC(curve = 'ROC')]
              )

auc = RocCallback(training_data = (X_train, y_train), validation_data = (X_test, y_test))
tfbd = tf.keras.callbacks.TensorBoard(log_dir = './logs/tf_fm', histogram_freq = 1)

#model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data = (X_test, y_test))#, callbacks = [auc])
model.fit(dataset, batch_size = 4, epochs = 300, validation_data = valset, callbacks = [tfbd])

