# Compare with raw EKG waveforms (Check the order)
i =5
fnm = df_pm.iloc[i, 3]
print(fnm)
ecg_df = pd.read_csv('/home/ubuntu/dr-you-ecg-20220420_mount/220927_SevMUSE_EKG_waveform/'+fnm+'.csv')
print(ecg_df)
print('\n')
print(X_pm[i])
