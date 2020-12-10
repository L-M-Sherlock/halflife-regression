import pandas as pd
import random

df_list = [pd.DataFrame(
    {
        'p': [1 if random.random() <= x/10 else 0 for _ in range(100000)],
        'pp': [x/10 - random.random() / 10 for _ in range(100000)]
    }
) for x in reversed(range(5, 10))]

df = pd.DataFrame(columns=['p', 'pp'])

for i in range(len(df_list)):
    df = pd.concat([df, df_list[i]])

df.to_csv('results/test.csv', index=False)

'''
完美算法
%%%%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%
AUC: 0.6663928842904214
%%%%%%%%%%%%%%%% metric %%%%%%%%%%%%%%%%
mean average error: 0.35079999999999684
root mean squared error: 0.41928510586472495
log metric: 0.5294325281488661

高估 0.05
%%%%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%
AUC: 0.6476550885254764
%%%%%%%%%%%%%%%% metric %%%%%%%%%%%%%%%%
mean average error: 0.3251499999999967
root mean squared error: 0.4214854683141463
log metric: 0.5439306418182921

低估 0.05
%%%%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%
AUC: 0.6623333333333334
%%%%%%%%%%%%%%%% metric %%%%%%%%%%%%%%%%
mean average error: 0.3756500000000054
root mean squared error: 0.42207819180810546
log metric: 0.5358708795030825

猜平均数
%%%%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%
AUC: 0.5
%%%%%%%%%%%%%%%% metric %%%%%%%%%%%%%%%%
mean average error: 0.377125
root mean squared error: 0.4354595273960601
log metric: 0.5670042468456393
'''