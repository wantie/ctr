import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from matplotlib import pyplot
from matplotlib import pylab
import time

from lr import LogisticRegressionClassifier
from fm import FactorialMachineClassifier

def plot_pr(auc_score, precision, recall, label=None):
  pylab.figure(num=None, figsize=(6, 5))
  pylab.xlim([0.0, 1.0])
  pylab.ylim([0.0, 1.0])
  pylab.xlabel('Recall')
  pylab.ylabel('Precision')
  pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))
  pylab.fill_between(recall, precision, alpha=0.5)
  pylab.grid(True, linestyle='-', color='0.75')
  pylab.plot(recall, precision, lw=1)
  pylab.show()

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


# 5 iter , get 0.55 auc, 10 iter get 0.59 auc
#model = LogisticRegressionClassifier(max_iter = 5, lr = 0.000001, alpha = 0.1, bs=1000000)

# 1 SGD iter for 0.61 auc
#model = LogisticRegressionClassifier(max_iter = 3, lr = 0.01, alpha = 0.1, bs=128)

# auc 0.67
#model = LogisticRegressionClassifier(max_iter = 3, lr = 0.01, alpha = 0.01, bs=128)


# auc 0.69 
#model = LogisticRegressionClassifier(max_iter = 3, lr = 0.01, alpha = 0.001, bs=64)

# auc 0.64
#model = LogisticRegressionClassifier(max_iter = 3, lr = 0.01, alpha = 0.001, bs=2)

# auc 0.7077, and stuck in log 0.40, both train and val
#model = LogisticRegressionClassifier(max_iter = 30, lr = 0.01, alpha = 0.000, bs=64)

# 0.7107, with sgd momentum
#model = LogisticRegressionClassifier(max_iter = 30, lr = 0.01, alpha = 0.001, bs=128)

# 0.7024, 
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, alpha = 0.001, bs=16, k = 4)

# 很快到0.70,但之后训练发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, alpha = 0.001, bs=8, k = 4)

# 发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, alpha = 0.001, bs=4, k = 4)

# 收敛，却太慢，且loss&auc不够
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00001, alpha = 0.001, bs=4, k = 4)

# SGD, bs为1，第一轮迭代效率很高达到0.6954， 第二轮到达0.699, 很容易发散, 即使在第一轮
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00001, alpha = 0.001, bs=1, k = 4)

# 依然后续发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.000006, alpha = 0.001, bs=1, k = 4)

# 前面learn ratio decay都为0.8， 这里为0.7， 依然发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.000006, alpha = 0.001, bs=1, k = 4)

# decay 0.7, 更小的方差0.01初始化参数,无正则, 提升到0.7049，继续难，
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00001, alpha = 0.001, bs=1, k = 4)

# 0.7040 观察是否是过拟合(或者正在过拟合)的问题, 太慢了, 
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00001, alpha = 0.001, bs=1, k = 4)

# decay 0.95, 历史新高，0.7083, 但第7轮开始发散, 发散较慢
# decay 0.90, 类似前面
# decay 0.85, 类似，发散略推迟
# decay 0.80, 忘记了
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00001, alpha = 0.001, bs=1, k = 4)

# 手动输入学习率, 太麻烦，且没什么用
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.00003, alpha = 0.001, bs=1, k = 4)

# 更改正则, 从0.001 增大为0.01, 忘记了
# 正则的影响，太大，训练不动，损失太高, 没有正则容易发散，适合的正则似乎能降低发散情况，且训练能正常进行
# 这个配置达到了新高，0.712, 超越了LR,  但后期还是缓慢发散了，猜测：一次项引起，学习率过高，二次项太大
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, lambda_1 = 0, lambda_2 = 0.0001, bs=1, k = 4)

# 增加一次项的正则, 发散一次,第二次训练，后期发散，但收敛较慢，到0.711
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, lambda_1 = 0.0001, lambda_2 = 0.0001, bs=1, k = 4)
# lambda 1继续降低, it 7,auc 0.707后发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, lambda_1 = 0.00001, lambda_2 = 0.0001, bs=1, k = 4)

# 从历史最高出发，增大二次项正则, 收敛，但损失太高，auc太低，0706...
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, lambda_1 = 0, lambda_2 = 0.01, bs=1, k = 4)
# lanmbda 2 调整, 比上次更小，比最优auc略大的正则，避免了发散，但效果不佳
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.0001, lambda_1 = 0, lambda_2 = 0.003, bs=1, k = 4)
# 固定正则，增大学习率, 极易发散
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.001, lambda_1 = 0, lambda_2 = 0.003, bs=1, k = 4)

# 固定正则，增大学习率, 极易发散
# 相应增加正则大小, 收敛，0.7112二次项的正则很关键！但二次参数越来越小，对模型的贡献越来越低(可惜没有查看全部参数)
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.01, bs=1, k = 4)
# 继续调整二次正则, 史上最好结果！！！0.71218, 正则可避免发散，也可增大学习率提升速度
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.006, bs=1, k = 4)
# 0.71453 在降低点看看: 新高！新高！！正确的方向，再次强调，正则可提效提速！
#model = FactorialMachineClassifier(max_iter = 50, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.0045, bs=1, k = 4)

# 前面实现错了！！！
# 0.7250
#model = FactorialMachineClassifier(max_iter = 100, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.0045, bs=16, k = 4)

# 0.7268
#model = FactorialMachineClassifier(max_iter = 100, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.0045, bs=16, k = 8)

model = FactorialMachineClassifier(max_iter = 100, lr = 0.001, lambda_1 = 0.000, lambda_2 = 0.0045, bs=16, k = 8)
# 

model.fit(X_train, y_train, X_test, y_test)
y_predict  = model.predict(X_test)
model.dump()

probs = model.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test, probs)
pyplot.plot(range(model.max_iter_), model.val_log_loss_, '-b', lw=1, label='SGD')

pyplot.show()
auc = auc(fpr, tpr)
print("auc", auc)
