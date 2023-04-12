# df_BBB =  # Label metadata table
# stecg_bbb =  # 12 leads EKG numpy (N, 5000, 12)
bbb_chartno_ls = df_BBB['CHART_NO'].to_list()
bbb_dt_ls = df_BBB['acq_datetime'].to_list()
bbb_ecgid_ls = df_BBB['ecg_id'].to_list()

# Compare with raw EKG waveforms (Check the order)
i = 0
fnm = df_BBB.iloc[i, 4]
print(fnm)
ecg_df = pd.read_csv('/home/ubuntu/dr-you-ecg-20220420_mount/220927_SevMUSE_EKG_waveform/'+fnm+'.csv')
print(ecg_df)
print(stecg_bbb[i])

#---

# Show 12 lead waveform
def show_12_wvform(i):
    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    plt.suptitle('['+ str(i) +']   '+'[ChartNo] '+ str(bbb_chartno_ls[i]) + '   [Date] ' + str(bbb_dt_ls[i])+ '   [ECG ID] ' + str(bbb_ecgid_ls[i]) + '     (Unknown BBB)', position=(0.5,0.9), fontweight='bold',fontsize=font_size*1.5)
    
    df_x = pd.DataFrame(stecg_bbb[i], columns=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL','aVF']) # i: serial number of BBB list (0-359)
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
    

# Save 12 lead waveform
def save_12_wvform(i):
    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    plt.suptitle('['+ str(i) +']   '+'[ChartNo] '+ str(bbb_chartno_ls[i]) + '   [Date] ' + str(bbb_dt_ls[i])+ '   [ECG ID] ' + str(bbb_ecgid_ls[i]) + '     (Unknown BBB)', position=(0.5,0.9), fontweight='bold',fontsize=font_size*1.5)
    
    df_x = pd.DataFrame(stecg_bbb[i], columns=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL','aVF']) # i: serial number of BBB list (0-359)
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
    filename = "unknown_BBB_{:03d}.png".format(i)
    fig.savefig(filepath + filename, dpi=fig.dpi)

filepath = bbb_dir + 'unknown_bbb/'

# Save All
for j in range(len(y_bbb)):
    save_12_wvform(j)
