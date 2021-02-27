import tensorflow as tf
from tensorflow import keras
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from callbacks import RocCallback
import numpy as np
import sys

# field_info : [ field_i_len, ]
class FmCrossLayer(keras.layers.Layer):
  def __init__(self, field_info, k = 8):
    super(FmCrossLayer, self).__init__()
    self.k = k
    self.field_info = field_info

  def call(self, inputs):
    square_of_sum = tf.square(tf.reduce_sum(inputs, axis = 1))
    sum_of_square = tf.reduce_sum(tf.square(inputs), axis = 1)
    return tf.reduce_sum(square_of_sum - sum_of_square, axis = 1, keepdims = True) * 0.5
    
class MLP(keras.layers.Layer):
  def __init__(self):
    super(MLP, self).__init__()
    self.layer_1 = keras.layers.Dense(units = 64, activation = keras.activations.relu)
    self.layer_2 = keras.layers.Dense(units = 64, activation = keras.activations.relu)
    self.layer_3 = keras.layers.Dense(units = 64, activation = keras.activations.relu)
    self.layer_4 = keras.layers.Dense(units = 1)

  def call(self, inputs):
    x = self.layer_1(inputs)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)

    return x

class DeepFm(keras.layers.Layer):
  def __init__(self, field_info, k = 8, stddev = 0.01):
    super(DeepFm, self).__init__()
    self.field_info = field_info
    self.feature_size = sum(field_info)
    self.k = k

    self.embedding_1 = keras.layers.Embedding(self.feature_size, 1)
    self.embedding = keras.layers.Embedding(sum(field_info), k)
    self.fm_cross = FmCrossLayer(field_info, k)
    self.bias = tf.Variable(tf.random.normal(shape = (), stddev = stddev), trainable = True)
    self.mlp = MLP()

  def call(self, inputs):
    linear = tf.reduce_sum(self.embedding_1(inputs), axis = 1) + self.bias
    embeddings = self.embedding(inputs)
    cross = self.fm_cross(embeddings)
    dense_embedding = tf.reshape(embeddings, shape = (-1, len(self.field_info) * self.k))
    dnn = self.mlp(dense_embedding)
    return tf.sigmoid(linear + cross + dnn)
 
def get_data():
  data = pd.read_csv("./data/train.csv.bak")#.head(10000)

  drop_columns = [
    'id', 'device_id','device_ip','device_model',
    'site_id', 'site_domain',
    'app_id', 'app_domain'
  ]

  print('shape before drop : ', data.shape)
  data.drop(drop_columns, axis=1, inplace = True)
  print('shape after drop : ', data.shape)
  #print(data.head(100))

  #for col in data.columns:
  #  print("value count of {0}\n{1}".format(col, data[col].value_counts()))


  dummy_dfs = []

  for col in data.columns :
    if col != 'click':
      dummy_dfs.append(pd.get_dummies(data[col], prefix=col))

  field_info = [df.shape[1] for df in dummy_dfs]
  X = pd.concat(dummy_dfs, axis = 1)
  print('shape of dummy train_set :', X.shape)
  where = X.apply(lambda x : np.where(x != 0)[0], axis = 1)
  columns = ['f{0}'.format(i) for i in range(len(field_info))]
  X = pd.DataFrame(where.to_list(), columns = columns)
  print('shape of dummy indice train_set :', X.shape)
  print(X.head(5))

  y = data['click']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=3)

  return X_train, y_train, X_test, y_test, field_info

X_train, y_train, X_test, y_test, field_info = get_data()

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(800000, reshuffle_each_iteration = False)
dataset = dataset.batch(64)
valset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10000)

inputs = keras.Input(shape = (X_train.shape[1],))

deep_fm = keras.Model(inputs, outputs = DeepFm(field_info, k = 32)(inputs))

#opt = tf.keras.optimizers.Adam()
opt = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
deep_fm.compile(optimizer = opt,
                loss = 'binary_crossentropy',
                metrics = [tf.keras.metrics.AUC(curve = 'ROC')]
)

tfbd = tf.keras.callbacks.TensorBoard(log_dir = './logs/deep_fm', histogram_freq = 1)


deep_fm.fit(dataset, epochs = 100, validation_data = valset, callbacks = [tfbd])
deep_fm.summary()
