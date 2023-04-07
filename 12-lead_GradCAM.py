# Calculate other 4 leads using sum of vectors
lead12_test = []
for i in range(len(y_test)):
    df_x = pd.DataFrame(X_test[i])
    lead_I = df_x[0].to_numpy()
    lead_II = df_x[1].to_numpy()

    lead_III = (np.subtract(lead_II, lead_I)*(0.5)).astype(int)
    lead_aVR = (np.add(lead_I, lead_II)*(-0.5)).astype(int)
    lead_aVL = (np.subtract(1.5*lead_I, 0.5*lead_II)*(0.5)).astype(int)
    lead_aVF = np.subtract(lead_II, 0.5*lead_I).astype(int)
    
    df_x['III'] = lead_III
    df_x['aVR'] = lead_aVR
    df_x['aVL'] = lead_aVL
    df_x['aVF'] = lead_aVF
    
    np_x = df_x.to_numpy()
    lead12_test.append(np_x)
    
stecg_test = np.array(lead12_test) #(1318, 5000, 12)
#np.save('/home/ubuntu/dr-you-ecg-20220420_mount/STEMI_JKL/Figures/st12ecg_test.npy', stecg_test)

# Draw Grad-CAM for 8 leads ECG and Shows 12 lead ECG
def ensemble_gradcam(models, layer_name, data):
    all_heatmaps = []
    for model in models:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        last_conv_layer_output, preds = grad_model(data)

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(data)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0))

        last_conv_layer_output = last_conv_layer_output[0]

        heatmap = last_conv_layer_output * pooled_grads
        heatmap = tf.reduce_mean(heatmap, axis=(1))
        heatmap = np.expand_dims(heatmap,0)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        all_heatmaps.append(heatmap.numpy())
        
    ensemble_heatmap = np.mean(all_heatmaps, axis=0)
        
    return ensemble_heatmap

models = de_model
layer_name = "conv1d_9"
label = ["Control", "Stemi"]

def gcperlead(idx):
    data = X_test[idx]
    data = np.expand_dims(data, 0)
    label = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(30,45))
    heatmap = ensemble_gradcam(models, layer_name, data)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((5000, 1))
    for i, lead in enumerate(label):
        axs[i].plot(stecg_test[idx][:, i], 'k')
        im = axs[i].imshow(np.expand_dims(heatmap,axis=2),cmap='Reds', aspect="auto", interpolation='nearest',extent=[0,5000,data.min(),data.max()],  alpha=0.8)
        axs[i].set_title('Lead '+str(lead), fontdict={'fontsize': 25})
        axs[i].tick_params(axis='both', which = 'major', labelsize=20)
        axs[i].margins(x=0)
    
    # create colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_position([.95, 0.17, 0.015, 1.2])

    # adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.4)
    plt.show()
    
 


# Show 12 lead waveform
def show_12_waveform(i):
    df_x = pd.DataFrame(X_test[i])
    lead_I = df_x[0].to_numpy()
    lead_II = df_x[1].to_numpy()
    lead_V1 = df_x[2].to_numpy()
    lead_V2 = df_x[3].to_numpy()
    lead_V3 = df_x[4].to_numpy()
    lead_V4 = df_x[5].to_numpy()
    lead_V5 = df_x[6].to_numpy()
    lead_V6 = df_x[7].to_numpy()
    
    lead_III = (np.subtract(lead_II, lead_I)*(0.5)).astype(int)
    lead_aVR = (np.add(lead_I, lead_II)*(-0.5)).astype(int)
    lead_aVL = (np.subtract(1.5*lead_I, 0.5*lead_II)*(0.5)).astype(int)
    lead_aVF = np.subtract(lead_II, 0.5*lead_I).astype(int)
    
    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    #plt.suptitle('[ChartNo] '+ str(wrong_pt_ls[i]) + '   [Date] ' + str(wrong_dt_ls[i]) + '     (True: STEMI / Predicted: Non-STEMI)', position=(0.5,0.95), fontweight='bold',fontsize=font_size*1.5)
    
    ax[0].plot(lead_I)
    ax[0].set_title('I',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[1].plot(lead_II)
    ax[1].set_title('II',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[2].plot(lead_V1)
    ax[2].set_title('V1',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[3].plot(lead_V2)
    ax[3].set_title('V2',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[4].plot(lead_V3)
    ax[4].set_title('V3',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[5].plot(lead_V4)
    ax[5].set_title('V4',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[6].plot(lead_V5)
    ax[6].set_title('V5',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[7].plot(lead_V6)
    ax[7].set_title('V6',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size
    
    ax[8].plot(lead_III)
    ax[8].set_title('III',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size
    
    ax[9].plot(lead_aVR)
    ax[9].set_title('aVR',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size

    ax[10].plot(lead_aVL)
    ax[10].set_title('aVL',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size
    
    ax[11].plot(lead_aVF)
    ax[11].set_title('aVF',fontweight="bold", size=font_size)
    plt.rcParams["figure.figsize"] = size
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


###############################################################################################################
## Initial Setting on Jupyter Notebook

# Import Modules
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv1D,Conv2D, Flatten
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import sklearn
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, recall_score, f1_score, balanced_accuracy_score, classification_report, roc_curve, precision_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import collections
from itertools import cycle

# Import Data
My_dir = '/home/ubuntu/dr-you-ecg-20220420_mount/STEMI_JKL/2023_Jan_testset/'
X_test, Y_test = np.load(My_dir+'x_test_0114.npy'), np.load(My_dir+'y_test_0114.npy')
print('X_test {}, Y_test {}'.format(X_test.shape, Y_test.shape)) #X_test (1318, 5000, 8), Y_test (1318,)

# Transform dataset to categorical (one-hot encoding)
y_test = tf.keras.utils.to_categorical(Y_test)

# Load saved model
mpath = '/home/ubuntu/Kyulee/ECG_2210/'
nets = 5
de_model = [0]*5
for i in [0,1,2,3,4]:
    de_model[i] = keras.models.load_model(mpath+'ami_model/BestModelSaved/221204_1643'+str(i+1)+'_bestmodel.h5')
    
# deep ensemble model TEST
def test_demodel(de_model, y_test, x_dataset):
    each_proba = []
    for m in de_model:
        proba = m.predict(x_dataset)
        each_proba.append(proba)
    results = np.zeros( (y_test.shape[0],2) )
    for p in each_proba:
        results += p
    de_proba = results / len(de_model)
    return de_proba
  
# Predict val and test data (Probability)
prob1_test = test_demodel(de_model, y_test, test_dataset)
threshold = 0.0768
def classify(proba):
    pred = (proba > threshold).astype(np.int64)
    return pred
pred_test = classify(prob1_test) 

(# Save probabilities)
df_test = pd.DataFrame({'Probs': prob1_test[:,1], 'Label': y_test[:,1]})
df_test.loc[df_test['Label']==1,'Label']='STEMI'
df_test.loc[df_test['Label']==0,'Label']='Control'
#df_test.to_csv(m_path+'/test_probabilities_of_STEMI_0406.csv', index=True)


#########################################################################################################
# Saving all figures (8 - leads)
for j in range(len(X_test)):
    data = X_test[j]
    data = np.expand_dims(data, 0)
    label = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(30,40))
    for i, lead in enumerate(label):
        heatmap = grad_cam(model, layer_name, data)
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((5000, 1))
        axs[i].plot(X_test[data_index][:, i], 'k')
        im = axs[i].imshow(np.expand_dims(heatmap,axis=2),cmap='Reds', aspect="auto", interpolation='nearest',extent=[0,5000,data.min(),data.max()],  alpha=0.8)
        axs[i].set_title('Lead '+str(lead), fontdict={'fontsize': 25})
        axs[i].tick_params(axis='both', which = 'major', labelsize=20)
        axs[i].margins(x=0)
    # create colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_position([.95, 0.17, 0.015, 1.2])

    # adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.4)
    plt.draw()
    
    # save it
    filename = "sing_lead_{:03d}.png".format(j+1)
    filepath =
    fig.savefig(filepath + filename, dpi=fig.dpi)


