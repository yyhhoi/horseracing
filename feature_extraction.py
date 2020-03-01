import pandas as pd
import numpy as np
# place: only entries that are numbers
# horse_no: exclude nan
# decla_weight: exclude '---'
# draw: exclude '---'
# time: exclude '---'
# odds: exclude '---'

# Index(['place', 'horse_no', 'horse_code', 'horse_name', 'horse_origin',
#        'jockey_name', 'trainer_name', 'act_weight', 'decla_weight', 'draw',
#        'lbw', 'time', 'odds', 'race_no', 'race_num', 'track_length', 'going',
#        'course', 'race_date', 'location'],




df_path = 'data/raw_df.csv'

raw_df = pd.read_csv(df_path, index_col=0)

raw_df.pop('horse_link')
raw_df.pop('lbw')

for col in raw_df.columns:
    nan_counts = raw_df[col].isna().sum()
    hythen_counts = np.sum(raw_df[col] == '---')
    print('%s: nan = %d, "---" = %d' % (col, nan_counts, hythen_counts))


import pdb
pdb.set_trace()


