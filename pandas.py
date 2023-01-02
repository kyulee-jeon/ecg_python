# change type of values in particular columns to datetime
def datetime_format(df, column):
    df[column] = pd.to_datetime(df[column], format = '%Y-%m-%d %H:%M:%S', errors='raise')

