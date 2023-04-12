# X_bbb =  # (N, 5000, 8)
# y_bbb =  # (N, 2)

lead12_bbb = []
for i in range(len(y_bbb)):
    df_x = pd.DataFrame(X_bbb[i])
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
    lead12_bbb.append(np_x)
    
stecg_bbb = np.array(lead12_bbb)
print(stecg_bbb.shape)

# Check the orders
np.all(stecg_bbb[:,:,:8]==X_bbb)
