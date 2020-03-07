import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
df = pd.read_csv('data/raw_df.csv', index_col=0)

df.pop('horse_link')


df["race_date"] = pd.to_datetime(df['race_date'], format="%Y/%m/%d")


year_delta = datetime.timedelta(days=365.25)

start_date = min(df['race_date'])
stop_date = max(df['race_date'])
years = []
recorder = []
while start_date < stop_date:
    end_date = start_date + year_delta
    selected_df = df[(df['race_date'] > start_date) & (df['race_date'] < end_date)]
    num_cases = selected_df.shape[0]
    recorder.append(num_cases)
    years.append(start_date.year)
    start_date = start_date + year_delta

plt.plot(years, recorder)
plt.show()


