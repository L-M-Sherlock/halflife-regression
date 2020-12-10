import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import roc_curve, auc


def mean(lst):
    # the average of a list
    return float(sum(lst)) / len(lst)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def rmse(l1, l2):
    # root mean squared error
    return math.sqrt(mean([math.pow(abs(l1[i] - l2[i]), 2) for i in range(len(l1))]))


def log_metric(l1, l2):
    # supermemo metric
    return mean([- math.log(1 - abs(l1[i] - l2[i])) for i in
                 range(len(l1))])


df = pd.read_csv("results/test.csv", sep=',')
df['p'] = df['p'].map(lambda x: round(x))

print("%%%%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%")

fpr, tpr, threshold = roc_curve(df['p'], df['pp'])
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print("%%%%%%%%%%%%%%%% metric %%%%%%%%%%%%%%%%")
print('mean average error:', mae(df['p'], df['pp']))
print('root mean squared error:', rmse(df['p'], df['pp']))
print('log metric:', log_metric(df['p'], df['pp']))
