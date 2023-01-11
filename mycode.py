# check malform of EKG_day
def check_dateform(df, col):
    nform_ls = []
    for i, yr in enumerate(df[str(col)]):
        yr = str(yr)
        if not re.match("\d{4}-\d{2}-\d{2}", yr):
            nform_ls.append(i)
    for i in nform_ls:
        print(df.iloc[i])


# change type of values in particular columns to datetime
def datetime_form(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col], format = '%Y-%m-%d %H:%M:%S', errors='raise')
    
def date_form(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col], format = '%Y-%m-%d', errors='raise')

def datetime_to_date(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col], format = '%Y-%m-%d %H:%M:%S', errors='raise')
    df['date'] = pd.DatetimeIndex(df[dt_col]).date    
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d', errors='raise')
    return df

df['year'] = pd.DatetimeIndex(df['datetime']).year
df['date'] = pd.DatetimeIndex(df['datetime']).date


def calibration_plot(true, probs, n_bins):
    # reliability diagram
    prob_true, prob_pred = calibration_curve(true[:,1], probs[:,1], n_bins=n_bins)

    # plot perfectly calibrated
    plt.plot([0,1], [0,1], linestyle='--')

    # plot model reliability
    plt.plot(prob_pred, prob_true, marker='.')
    plt.title('Calibration Plot (bins: {0:d})' ''.format(n_bins))
    plt.show()

# sing_proba 분포
num = np.arange(len(sing_probs))
proba_sort = np.sort(sing_probs[:,1])
plt.plot(num, proba_sort)

# deep ensemble model TEST
def test_demodel(de_model, x_dataset, cut_off):
    each_proba = []
    for m in de_model:
        proba = m.predict(x_dataset)
        each_proba.append(proba)
    results = np.zeros( (x_dataset.shape[0],2) )
    for p in each_proba:
        results += p
    de_proba = results / len(de_model)
    #de_pred = (de_proba > cut_off).astype(np.int64)
    return de_proba

# list 해당하는 dataframe 추출
df[df['col'].isin(ls)]
# not isin
df[~df['col'].isin(ls)]

# sorting
df.sort_values(by='col')

# check and delete duplicated
df[df['col'].duplicated()]
df.drop_duplicates(['col'], keep='first')

# 중복 제거
def drop_dup(df, subset, sort):
    df = df.drop_duplicates(subset=subset)
    df = df.sort_values(by=[sort], ascending=True)
    df = df.reset_index(drop=True)
    return df

# 특정 행 제거
df.drop(index)

# (1) 날짜 차이 계산
def diff_days(df, dt_M, dt_m):
    df['dt_M'] = pd.DatetimeIndex(df_trop[dt_M]).date
    df['dt_m'] = pd.DatetimeIndex(df_trop[dt_m]).date
    
    df['dt_M'] = pd.to_datetime(df['dt_M'], format = '%Y-%m-%d', errors='raise')
    df['dt_m'] = pd.to_datetime(df['dt_m'], format = '%Y-%m-%d', errors='raise')
    
    for i in range(len(df_trop)):
        diff = df.loc[i, 'dt_M']- df.loc[i, 'dt_m']
        diff = diff.days
        df.loc[i, 'distance'] = diff
    df['distance'] = df['distance'].abs()
    return df

# (2) 차이 최소 groupby
def diff_min(df, criteria_1, criteria_2, min_col):
    df_min = df.loc[df.groupby([criteria_1, criteria_2])[min_col].idxmin()]
    df_min = df_min.sort_values(by=[criteria_1], ascending = True)
    df_min = df_min.reset_index(drop=True)
    return df_min

# (1+2) 날짜 차이 최소
def diff_days_min(df, dt_M, dt_m, criteria_1, criteria_2):
    df['dt_M'] = pd.DatetimeIndex(df_trop[dt_M]).date
    df['dt_m'] = pd.DatetimeIndex(df_trop[dt_m]).date
    
    df['dt_M'] = pd.to_datetime(df['dt_M'], format = '%Y-%m-%d', errors='raise')
    df['dt_m'] = pd.to_datetime(df['dt_m'], format = '%Y-%m-%d', errors='raise')
    
    for i in range(len(df_trop)):
        diff = df.loc[i, 'dt_M']- df.loc[i, 'dt_m']
        diff = diff.days
        df.loc[i, 'distance'] = diff
    df['distance'] = df['distance'].abs()
    df_min = df.loc[df.groupby([criteria_1, criteria_2])['distance'].idxmin()]
    df_min = df_min.sort_values(by=[criteria_1], ascending = True)
    df_min = df_min.reset_index(drop=True)
    df_min = df_min.drop(['dt_M', 'dt_m'], axis=1)
    return df_min

print('{} Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n'.format(
            name, total, pos, 100 * pos / total, neg, 100 * neg / total))
<<<<<<< HEAD

# 열 type 변경
df['col'] = df['col'].astype(int)
=======
>>>>>>> 9a995cd8e038a056d9001a50e4c8a8bb82a032c0
