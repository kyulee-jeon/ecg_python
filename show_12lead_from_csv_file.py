def show_12_waveform_fromDF(df, i):
    ecg_folder = '/home/ubuntu/dr-you-ecg-20220420_mount/220927_SevMUSE_EKG_waveform/'
    fname = df.iloc[i, 4]  # df 4열은 ecg_id (e.g. df_train, df_test, etc)
    df_x = pd.read_csv(ecg_folder+fname+'.csv')
    dc_x = df_x.to_dict('list')
    leads = list(dc_x.keys())

    size = (40,45)
    font_size = 25
    fig, ax = plt.subplots(12, 1)   
    plt.suptitle('['+ str(i) +']', position=(0.5,0.9), fontweight='bold',fontsize=font_size*1.5)
    
    for j in range(12):
        lead_nm = leads[j]
        lead_ecg = np.array(dc_x[lead_nm])
        ax[j].plot(lead_ecg)
        ax[j].set_title(lead_nm, fontweight="bold", size=font_size)
        plt.rcParams["figure.figsize"] = size

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
