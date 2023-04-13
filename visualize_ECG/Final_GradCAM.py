#df_test.to_csv(m_path+'/test_probabilities_of_STEMI_0413.csv', index=True)

# Change the Order of Leads 
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
    
    df_x.columns = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL','aVF']
    df_ecg = df_x[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    
    
    np_ecg = df_ecg.to_numpy()
    lead12_test.append(np_ecg)

# Ensemble
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
    
    
################################################################################################################    
# Visualize (basic version)
def gcperlead(idx):
    data = X_test[idx]
    data = np.expand_dims(data, 0)
    label = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    heatmap = ensemble_gradcam(models, layer_name, data)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((5000, 1))
    
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(30,45))
    plt.suptitle('{}\n\n[ True label :  {} ,  Prediction :  {}  ({:.1%})  ]'.format(idx, true_ls[idx], pred_ls[idx], prob_ls[idx]), position = (0.5, 0.93), fontweight='bold', fontsize= 25)
    
    for i, lead in enumerate(label):
        axs[i].plot(stecg_test[idx][:, i], 'k')
        im = axs[i].imshow(np.expand_dims(heatmap,axis=2),cmap='Reds', aspect="auto", interpolation='nearest',extent=[0,5000,data.min(),data.max()],  alpha=0.8)
        axs[i].set_title('Lead '+str(lead), fontdict={'fontsize': 23})
        axs[i].tick_params(axis='both', which = 'major', labelsize=15)
        axs[i].margins(x=0)
    
    # create colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_position([.95, 0.17, 0.015, 1.2])

    # adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5)
    plt.show()


# Other Color Distribution (Plasma Colors)
def gcperlead_v2(idx):
    data = X_test[idx]
    data = np.expand_dims(data, 0)
    label = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    heatmap = ensemble_gradcam(models, layer_name, data)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((5000, 1))
    
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(30,45))
    plt.suptitle('{}\n\n[ True label :  {} ,  Prediction :  {}  ({:.1%})  ]'.format(idx, true_ls[idx], pred_ls[idx], prob_ls[idx]), position = (0.5, 0.93), fontweight='bold', fontsize= 25)
    
    for i, lead in enumerate(label):
        axs[i].plot(stecg_test[idx][:, i], 'k')
        im = axs[i].imshow(np.expand_dims(heatmap,axis=2),cmap='plasma', aspect="auto", interpolation='nearest',extent=[0,5000,data.min(),data.max()],  alpha=0.8)
        axs[i].set_title('Lead '+str(lead), fontdict={'fontsize': 23})
        axs[i].tick_params(axis='both', which = 'major', labelsize=15)
        axs[i].margins(x=0)
    
    # create colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_position([.95, 0.17, 0.015, 1.2])

    # adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5)
    plt.show()


# Plasma Colors and Show Confidence
for i in range(len(df_test)):
    prob = df_test.iloc[i, 0]
    if prob > threshold: # 0.0768
        df_test.loc[i, 'Conf'] = prob
    else:
        df_test.loc[i, 'Conf'] = 1-prob

true_ls = df_test['Label'].to_list()
pred_ls = df_test['Pred'].to_list()
conf_ls = df_test['Conf'].to_list()

def gcperlead_v3(idx):
    data = X_test[idx]
    data = np.expand_dims(data, 0)
    label = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    heatmap = ensemble_gradcam(models, layer_name, data)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((5000, 1))
    
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=(30,45))
    plt.suptitle('{}\n\n[ True label :  {} ,  Prediction :  {}  ({:.1%})  ]'.format(idx, true_ls[idx], pred_ls[idx], conf_ls[idx]), position = (0.5, 0.93), fontweight='bold', fontsize= 25)
    
    for i, lead in enumerate(label):
        axs[i].plot(stecg_test[idx][:, i], 'k')
        im = axs[i].imshow(np.expand_dims(heatmap,axis=2),cmap='plasma', aspect="auto", interpolation='nearest',extent=[0,5000,data.min(),data.max()],  alpha=0.8)
        axs[i].set_title('Lead '+str(lead), fontdict={'fontsize': 23})
        axs[i].tick_params(axis='both', which = 'major', labelsize=15)
        axs[i].margins(x=0)
    
    # create colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_position([.95, 0.17, 0.015, 1.2])

    # adjust spacing between subplots
    fig.subplots_adjust(hspace = 0.5)
    plt.show()
    
    
  ###############################################################################################
  tn_ls = df_test[(df_test['Label']=='Control') & (df_test['Label']==df_test['Pred'])].index
  fp_ls = df_test[(df_test['Label']=='Control') & (df_test['Label']!=df_test['Pred'])].index
  
  for i in fp_ls[10:20]:
    gcperlead(i)
 
