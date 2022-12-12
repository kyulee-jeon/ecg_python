import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(labels, predictions, p=0.5):
    cf_matrix = confusion_matrix(labels, predictions > p)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["({0:.2%})".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
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