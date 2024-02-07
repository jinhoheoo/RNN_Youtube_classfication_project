import pandas as pd
import glob
import datetime

last_data = []


data_paths = glob.glob('./predict_data/data_*')


print(last_data)

df = pd.DataFrame()
for path in data_paths:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df = pd.concat([df, df_temp])

if not df.empty:
    # Check if 'keyword' column is present in the DataFrame
    if 'keyword' in df.columns:
        print(df['keyword'].value_counts())
    else:
        print("Column 'keyword' not found in the DataFrame.")

    df.info()
    df.drop_duplicates(subset='titles',inplace=True)
    df.info()

    df.to_csv('./Youtube_titles_{}.csv'.format(
        datetime.datetime.now().strftime('%Y%m%d')), index=False)
else:
    print("DataFrame is empty. No CSV file generated.")