# confusion matrix
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, recall_score, f1_score, balanced_accuracy_score, classification_report, roc_curve, precision_score
def matrix(y_true, y_pred, y_proba):
    conf_mat = confusion_matrix(y_true[:,1], y_pred[:,1])
    TP = conf_mat[1][1]
    TN = conf_mat[0][0]
    FN = conf_mat[1][0]
    FP = conf_mat[0][1]

    acc = round((TP+TN)/(TP+TN+FP+FN), 4)
    sens = round(TP/(TP+FN), 4)
    spec = round(TN/(TN+FP), 4)
    ppv = round(TP/ (TP+FP), 4)
    npv = round(TN/ (TN+FN), 4)
    auroc = round(roc_auc_score(y_true, y_proba), 4)
    precision = round(precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)), 4)
    recall = round(recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)), 4)
    f1score = round(f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)), 4)
    ap = round(average_precision_score(y_true, y_pred), 4)

    print(conf_mat, "\n")
    print("Accuracy", acc)
    print("Sensitivity", sens)
    print("Specificity", spec)
    print("PPV", ppv)
    print("NPV", npv)
    print("AUROC", auroc)
    print("Average Precision", ap)
    print('Precision', precision)
    print("Recall", recall)
    print("F1",  f1score)
    print(classification_report(y_true, y_pred))
    
    # ROC & AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_true.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red'])
    for i, color in zip(range(y_true.shape[1]), colors):
        if i == 0:
            pass
        else:
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve (area = {0:0.2f})' ''.format(roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], lw=1.5, color='black', linestyle='dotted', label = 'baseline')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

label = {'0': 'Not-STEMI', '1':'STEMI'}

# pr curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

def prc(y_true, y_pred, y_proba):
    # PR & AUC
    prec = dict()
    rec = dict()
    pr_auc = dict()
    for i in range(y_true.shape[1]):
        prec[i], rec[i], _ = sklearn.metrics.precision_recall_curve(y_true[:,i], y_proba[:, i])
        pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=prec[i], recall=rec[i])
    pr_display.plot(label = 'Precision-Recall curve (area = {0:0.2f})' ''.format(average_precision_score(y_true, y_pred)))
    plt.show()

label = {'0': 'Non-STEMI', '1':'STEMI'}

def plot_cm(labels, predictions, p):
    cf_matrix = confusion_matrix(labels, predictions > p)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages_pos = ["({0:.2%})".format(value) for value in cf_matrix.flatten()[0:2]/np.sum(cf_matrix.flatten()[0:2])]
    group_percentages_neg = ["({0:.2%})".format(value) for value in cf_matrix.flatten()[2:4]/np.sum(cf_matrix.flatten()[2:4])]
    group_percentages = group_percentages_pos + group_percentages_neg
    categories = ['Not-STEMI', 'STEMI']
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    #plt.figure(figsize=(5,5))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',xticklabels=categories,yticklabels=categories, annot_kws={"size": 12})
    plt.title('Confusion Matrix (cut off = {:.2f})'.format(p), fontsize= 14)
    plt.ylabel('Actual label', fontsize= 12)
    plt.xlabel('Predicted label', fontsize= 12)

    print('True Negatives: ', cf_matrix[0][0])
    print('False Positives: ', cf_matrix[0][1])
    print('False Negatives: ', cf_matrix[1][0])
    print('True Positives: ', cf_matrix[1][1])
    print('Total STEMI: ', np.sum(cf_matrix[1]))


# print all
def result_report(y_test, y_pred, y_proba, threshold):
    print(matrix(y_test, y_pred, y_proba))
    print(prc(y_test, y_pred, y_proba))
    print(plot_cm(y_test[:,1], y_pred[:,1], threshold))


# calibration plot
import sklearn
from sklearn.calibration import calibration_curve
def calibration_plot(true, probs, n_bins):
    # reliability diagram
    prob_true, prob_pred = calibration_curve(true[:,1], probs[:,1], n_bins=n_bins)

    # plot perfectly calibrated
    plt.plot([0,1], [0,1], linestyle='--')

    # plot model reliability
    plt.plot(prob_pred, prob_true, marker='.')
    plt.title('Calibration Plot (bins: {0:d})' ''.format(n_bins))
    plt.show()


## Uncertainty

# Deep Ensembles
de_entropy_list = []
for i in range(len(dval_proba)):
    pq = dval_proba[i] * y_val[i] #정답에 대한 예측 확률 사용하는거 맞는지 확인
    index = np.argmax(pq)
    p = pq[index]
    entropy = -p*math.log(p) - (1-p)*math.log(1-p+ 0.00000000001)
    de_entropy_list.append(entropy)

# Single 
sing_entropy_list = []
for i in range(len(sval_proba)):
    pq = sval_proba[i] * y_val[i] #정답에 대한 예측 확률 사용하는거 맞는지 확인
    index = np.argmax(pq)
    p = pq[index]
    entropy = -p*math.log(p) - (1-p)*math.log(1-p+ 0.00000000001)
    sing_entropy_list.append(entropy)

# plot distribution
sns.kdeplot(de_entropy_list, label='Deep Ensembles', color='r')
sns.kdeplot(sing_entropy_list, label='Single', color='g')
plt.xlabel('Validation Entropy Values')
plt.legend()
plt.show()

def uc_grouping(entropy_list, cutoff):
    high_idx = []
    low_idx = []
    for idx, entropy in enumerate(entropy_list):
        if entropy >= cutoff:
            high_idx.append(idx)
        else:
            low_idx.append(idx)
            
    return high_idx, low_idx

def get_co(list_a, percent):
    '''
    list_a: [int, int, ...] 
    num: int, 추출하고 싶은 개수 
    '''
    tmp = list_a.copy()
    tmp.sort()
    num = round(len(y_test)* percent*0.01)
    top_num = tmp[-num:]
    cutoff = round(min(top_num),4)
    
    return cutoff

co_05 = get_co(de_entropy_list, 5)


def compare_uncertainty(entropy_list, y_true, y_proba, y_pred, cutoff, threshold):
    high, low = uc_grouping(entropy_list, cutoff=cutoff)
    
    true_high = y_true[high]
    pred_high = y_pred[high]
    proba_high = y_proba[high]

    true_low = y_true[low]
    pred_low = y_pred[low]
    proba_low = y_proba[low]
    
    #print("Uncertainty High: %.2f%%" % (len(high) / len(entropy_list) * 100.0)), print(result_report(true_high, pred_high, proba_high, threshold))
    print("Uncertainty Low: %.2f%%" % (len(low) / len(entropy_list) * 100.0)), print(result_report(true_low, pred_low, proba_low, threshold))

def high_uncertainty(entropy_list, y_true, y_proba, y_pred, cutoff, threshold):
    high, low = uc_grouping(entropy_list, cutoff=cutoff)
    true_high = y_true[high]
    pred_high = y_pred[high]
    proba_high = y_proba[high]
    print("Uncertainty High: %.2f%%" % (len(high) / len(entropy_list) * 100.0))
    print(result_report(true_high, pred_high, proba_high, threshold))

def low_uncertainty(entropy_list, y_true, y_proba, y_pred, cutoff, threshold):
    high, low = uc_grouping(entropy_list, cutoff=cutoff)
    true_low = y_true[low]
    pred_low = y_pred[low]
    proba_low = y_proba[low]
    print("Uncertainty Low: %.2f%%" % (len(low) / len(entropy_list) * 100.0))
    print(result_report(true_low, pred_low, proba_low, threshold))

high_uncertainty(de_entropy_list, y_val, dval_proba, dval_pred, co_15, dval_yi)
