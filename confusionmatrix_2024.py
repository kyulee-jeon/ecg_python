from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def drawing_cm(y_true, y_pred):
  cf_matrix = confusion_matrix(y_true, y_pred)
  group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages_neg = ["({0:.2%})".format(value) for value in cf_matrix.flatten()[0:2]/np.sum(cf_matrix.flatten()[0:2])]
  group_percentages_pos = ["({0:.2%})".format(value) for value in cf_matrix.flatten()[2:4]/np.sum(cf_matrix.flatten()[2:4])]
  group_percentages = group_percentages_neg + group_percentages_pos
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
  labels = np.asarray(labels).reshape(2,2)

  # setting
  x_title = ['Not-STEMI', 'STEMI']
  y_title = ['Not-STEMI', 'STEMI']
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = 'Blues',
              xticklabels = x_title,
              yticklabels = y_title,
              annot_kws = {"size": 12})
  plt.title('Confusion Matrix (cut off = 0.08)', fontsize= 14)
  plt.ylabel('Actual label', fontsize= 12)
  plt.xlabel('AI-drived prediction', fontsize= 12)
  plt.show()
