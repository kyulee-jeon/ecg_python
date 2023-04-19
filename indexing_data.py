def indexing_data(target_df, df, X_data, y_data):
    target_id = target_df['ecg_id'].to_list()
    target_index = df[df['ecg_id'].isin(target_id)].index
    X_target = X_data[target_index]
    y_target = y_data[target_index]
    print(X_target.shape, y_target.shape)
    return X_target, y_target
  
X_pm1, y_pm1 = indexing_data(df_pm, df_train, X_train, y_train)
X_pm2, y_pm2 = indexing_data(df_pm, df_test, X_test, y_test)

X_pm = np.concatenate((X_pm1, X_pm2))
y_pm = np.concatenate((y_pm1, y_pm2))
