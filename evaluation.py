import matplotlib.pyplot as plt
import numpy as np
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


def load_brier(predictions, real, bins=20):
    counts = np.zeros(bins)
    correct = np.zeros(bins)
    prediction = np.zeros(bins)
    for p, r in zip(predictions, real):
        bin = min(int(p * bins), bins - 1)
        counts[bin] += 1
        correct[bin] += r
        prediction[bin] += p
    prediction_means = prediction / counts
    prediction_means[np.isnan(prediction_means)] = ((np.arange(bins) + 0.5) / bins)[np.isnan(prediction_means)]
    correct_means = correct / counts
    # correct_means[np.isnan(correct_means)] = 0
    size = len(predictions)
    answer_mean = sum(correct) / size
    return {
        "reliability": sum(counts * (correct_means - prediction_means) ** 2) / size,
        "resolution": sum(counts * (correct_means - answer_mean) ** 2) / size,
        "uncertainty": answer_mean * (1 - answer_mean),
        "detail": {
            "bin_count": bins,
            "bin_counts": list(counts),
            "bin_prediction_means": list(prediction_means),
            "bin_correct_means": list(correct_means),
        }
    }


def plot_brier(predictions, real, bins=20):
    brier = load_brier(predictions, real, bins=bins)
    plt.figure()
    plt.plot((0, 1), (0, 1), label='Optimal average observation')
    plt.plot(brier['detail']['bin_prediction_means'], brier['detail']['bin_correct_means'],  '*', label='Average observation',)
    bin_count = brier['detail']['bin_count']
    counts = np.array(brier['detail']['bin_counts'])
    bins = (np.arange(bin_count) + 0.5) / bin_count
    plt.legend(loc='upper center')
    plt.xlabel('Prediction')
    plt.ylabel('Observeation')
    plt.twinx()
    plt.ylabel('Number of predictions')
    plt.bar(bins, counts, width=(0.5 / bin_count), alpha=0.5, label='Number of predictions')
    plt.legend(loc='lower center')

if __name__ == "__main__":
    # df = pd.read_csv("con_test_1214-2.csv", sep=',')
    # df = pd.read_csv('results/lr.con_test_1214-2.preds', sep='\t')
    # df = pd.read_csv('results/lr.cr.con_test.preds', sep='\t')
    # df = pd.read_csv('results/anki.con_test.preds', sep='\t')
    # df = pd.read_csv('results/leitner.con_test.preds', sep='\t')
    # df = pd.read_csv('results/hlr.cr.con_test.preds', sep='\t')
    df = pd.read_csv('results/test.csv', sep=',')
    df['p'] = df['p'].map(lambda x: round(x))

    plot_brier(df['pp'], df['p'], 100)
    plt.show()

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
    # print('mean average error:', mae(df['p'], df['pp']))
    print('root mean squared error:', rmse(df['p'], df['pp']))
    print('log metric:', log_metric(df['p'], df['pp']))
