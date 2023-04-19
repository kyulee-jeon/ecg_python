# X_data =  # (N, 5000, 8)
# y_data =  # (N, 2)

def cal_12lead_shape(X_data, y_data):
    lead12 = []
    for i in range(len(y_data)):
        df_x = pd.DataFrame(X_data[i])
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
        lead12.append(np_x)

    stecg = np.array(lead12)
    print(stecg.shape)
    return stecg


def cal_12lead(X_data, y_data):
    lead12 = []
    for i in range(len(y_data)):
        df_x = pd.DataFrame(X_data[i])
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
        lead12.append(np_x)

    stecg = np.array(lead12)
    #print(stecg.shape)
    return stecg

def make_ls(df):
    chartno_ls = df['CHART_NO'].to_list()
    dt_ls = df['acq_datetime'].to_list()
    ecgid_ls = df['ecg_id'].to_list()
    return chartno_ls, dt_ls, ecgid_ls

# Show 12 lead waveform
def show_12_wvform(df, X_data, y_data, i):
    stecg = cal_12lead(X_data, y_data)
    chartno_ls, dt_ls, ecgid_ls = make_ls(df)
    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    Title = input('Title: ')
    plt.suptitle('['+ str(i) +']   '+'[ChartNo] '+ str(chartno_ls[i]) + '   [Date] ' + str(dt_ls[i])+ '   [ECG ID] ' + str(ecgid_ls[i]) + '     ('+str(Title)+')', position=(0.5,0.9), fontweight='bold',fontsize=font_size*1.5)
    
    df_x = pd.DataFrame(stecg[i], columns=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL','aVF']) # i: serial number of BBB list (0-359)
    df_x = df_x[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    lead_nm = df_x.columns
    for j, nm in enumerate(lead_nm):
        lead = df_x[nm].to_numpy()
        ax[j].plot(lead)
        ax[j].set_title(nm, fontweight="bold", size=font_size)
        plt.rcParams["figure.figsize"] = size
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    
show_12_wvform(df_test, X_test, y_test, 3)

#################################################################################################################
# Save 12 lead waveform
def save_12_wvform(directory, Title, df, X_data, y_data, i):
    stecg = cal_12lead(X_data, y_data)
    chartno_ls, dt_ls, ecgid_ls = make_ls(df)
    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    plt.suptitle('['+ str(i) +']   '+'[ChartNo] '+ str(chartno_ls[i]) + '   [Date] ' + str(dt_ls[i])+ '   [ECG ID] ' + str(ecgid_ls[i]) + '     ('+str(Title)+')', position=(0.5,0.9), fontweight='bold',fontsize=font_size*1.5)
    
    df_x = pd.DataFrame(stecg[i], columns=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL','aVF']) # i: serial number of BBB list (0-359)
    df_x = df_x[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    lead_nm = df_x.columns
    for j, nm in enumerate(lead_nm):
        lead = df_x[nm].to_numpy()
        ax[j].plot(lead)
        ax[j].set_title(nm, fontweight="bold", size=font_size)
        plt.rcParams["figure.figsize"] = size
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    # save it
    filepath = directory + str(Title)
    filename = "{}_{:03d}.png".format(str(Title), i)
    fig.savefig(filepath + filename, dpi=fig.dpi)
    
    
   



