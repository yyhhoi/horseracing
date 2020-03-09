import pandas as pd
import numpy as np
from lib.features import time2sec, calc_cul_winrates, EloEstimator, DataEncoder, columns_order, feature_cols, target_cols, linear_cols, cat_cols, metainfo_cols, DataGenerator
from lib.utils import find_missing
import pickle
import os
import random


# ------------------------------------------------------------------------------------------------------------------
# Index(['place', 'horse_link' (deleted) , 'horse_no' (deleted), 'horse_code', 'horse_name', 'horse_origin',
#        'jockey_name', 'trainer_name', 'act_weight', 'decla_weight', 'draw',
#        'lbw' (deleted), 'time', 'odds', 'race_no' (deleted), 'race_num', 'track_length', 'going',
#        'course', 'race_date', 'location'],


# Load, pop, cleaning and convert dtype
df_path = 'data/intermediate_storage/raw_df.csv'
raw_df = pd.read_csv(df_path, index_col=0)
raw_df.pop('horse_link')
raw_df.pop('lbw')
raw_df.pop('race_no')
raw_df.pop('horse_no')

# General cleaning (nan and '---')
for col in raw_df.columns:
    # Clean nan
    non_nan_mask = raw_df[col].isna() == False
    raw_df = raw_df[non_nan_mask]

    # Clean '---'
    hythen_mask = raw_df[col] == '---'
    raw_df = raw_df[hythen_mask == False]

# Clean place
non_place_pattern = '[A-Za-z]'
raw_df = raw_df[raw_df.place.str.contains(non_place_pattern) == False]
raw_df['place'] = raw_df['place'].astype(float)  # First to float, in order to parse strings like "14.0"

# Convert datatype
raw_df['act_weight'] = raw_df['act_weight'].astype(float)
raw_df['decla_weight'] = raw_df['decla_weight'].astype(float)
raw_df['draw'] = raw_df['draw'].astype(float)
raw_df['odds'] = raw_df['odds'].astype(float)
raw_df['race_num'] = raw_df['race_num'].astype(int)
raw_df['track_length'] = raw_df['track_length'].astype(int)

# Convert date and time
raw_df["race_date"] = pd.to_datetime(raw_df['race_date'], format="%Y/%m/%d")
raw_df["time"] = pd.to_datetime(raw_df['time'], format='%M:%S.%f').dt.time
raw_df['time'] = raw_df['time'].apply(time2sec)
raw_df = raw_df[raw_df['time'] > 0]

# raw_df = raw_df[raw_df['race_date'] > pd.Timestamp(2018, 1, 1)]

# Update age, weight difference
raw_df['age'] = np.nan
raw_df['time_since_last'] = np.nan
raw_df['weight_diff'] = np.nan

all_uni_horses = raw_df.horse_code.unique()
for i, uni_horse in enumerate(all_uni_horses):
    print('\rCalculate age %d/%d' % (i, len(all_uni_horses)), flush=True, end="")
    horse_df = raw_df[raw_df['horse_code'] == uni_horse]
    horse_dates = horse_df["race_date"]
    horse_indexes = horse_dates.index
    mindate = min(horse_dates)
    ages_series = (horse_dates.reset_index(drop=True) - mindate).apply(lambda x: x.days / 365.25)
    raw_df.loc[horse_indexes, 'age'] = list(ages_series)

    horse_df_sorted = horse_df.sort_values('race_date', ascending=False)
    sorted_indexes = horse_df_sorted.index

    # Time since last race
    horse_dates_sorted = horse_df_sorted['race_date']
    dates_plus = horse_dates_sorted[0:-1].reset_index(drop=True)
    dates_minus = horse_dates_sorted[1:].reset_index(drop=True)
    datesdiff = (dates_plus.reset_index(drop=True) - dates_minus.reset_index(drop=True)).apply(lambda x: x.days)
    complete_datesdiff = list(datesdiff) + [np.nan]
    raw_df.loc[sorted_indexes, 'time_since_last'] = complete_datesdiff

    # Weight difference
    horse_weights_sorted = horse_df_sorted['act_weight']
    weights_plus = horse_weights_sorted[0:-1]
    weights_minus = horse_weights_sorted[1:]
    weightsdiff = (weights_plus.reset_index(drop=True) - weights_minus.reset_index(drop=True))
    complete_weightsdiff = list(weightsdiff) + [np.nan]
    raw_df.loc[sorted_indexes, 'weight_diff'] = complete_weightsdiff

print()

# Mean imputation
mean_time_since_last = np.nanmean(raw_df['time_since_last'])
raw_df.loc[raw_df['time_since_last'].isna(), 'time_since_last'] = mean_time_since_last
mean_weight_diff = np.nanmean(raw_df['weight_diff'])
raw_df.loc[raw_df['weight_diff'].isna(), 'weight_diff'] = mean_weight_diff

# Cumulative wining times, wining rates and race times
for key in ['horse', 'jockey', 'trainer']:
    raw_df = calc_cul_winrates(key, raw_df)

# Intermediate storage
save_dir = 'data/intermediate_storage/stage3_features_engineering'

# ELO calculation
elo = EloEstimator(df=raw_df)
for key in ['horse', 'jockey', 'trainer']:
    elo.set_params(K=10, D=400)
    _, (_, rating_df) = elo.score(key, record_result=False, update_df=True)
    rating_df.to_csv(os.path.join(save_dir, '%s_rating.csv' % key))
raw_df = elo.df
raw_df.to_csv(os.path.join(save_dir, 'after_elo_df.csv'))

# Load Intermediate storage
save_dir = 'data/intermediate_storage/stage3_features_engineering'
raw_df = pd.read_csv(os.path.join(save_dir, 'after_elo_df.csv'), index_col=0)
raw_df['race_date'] = pd.to_datetime(raw_df['race_date'], format='%Y-%m-%d')

# Normalization and encoding
save_dir = 'data/training_dataframes'
# Categories: place horse_origin, track_length, going, course, location
normer = DataEncoder(raw_df)

# normer.create_category_tables()
# lsp_df = normer.create_linear_scale_params()

cat_dict = normer.load_category_tables()
linear_dict = normer.load_linear_scale_params()
encoding_df, scaled_df = normer.transform2encodings(vec=False)
encoding_df_vec, _ = normer.transform2encodings(vec=True)
with open(os.path.join(save_dir, 'encoding_df.pickle'), 'wb') as fh:
    pickle.dump(encoding_df, fh)
with open(os.path.join(save_dir, 'encoding_df_vec.pickle'), 'wb') as fh:
    pickle.dump(encoding_df_vec, fh)
with open(os.path.join(save_dir, 'scaled_df.pickle'), 'wb') as fh:
    pickle.dump(scaled_df, fh)
pop_cols = ['horse_name', 'jockey_name', 'trainer_name', 'horse_code']
for popcol in pop_cols:
    encoding_df_vec.pop(popcol)
encoding_df_vec = encoding_df_vec[columns_order]
metainfo_df = encoding_df_vec[metainfo_cols]
cat_features_df = encoding_df_vec[cat_cols].apply(np.concatenate, axis=1)
linear_features_df = encoding_df_vec[linear_cols].apply(np.array, axis=1)
features_df = pd.concat([cat_features_df, linear_features_df], axis=1).apply(np.concatenate, axis=1)
target_df = encoding_df_vec[target_cols]
df_all = pd.concat([metainfo_df, features_df, target_df], axis=1)
df_all.columns = metainfo_cols + ['features'] + target_cols
with open(os.path.join(save_dir, 'processed_df.pickle'), 'wb') as fh:
    pickle.dump(df_all, fh)

# data generator

df_path = os.path.join('data/training_dataframes', 'processed_df.pickle')




import time
datagen = DataGenerator(df_path=df_path, m=512)

for (x_train, y_train), (x_test, y_test) in datagen.iterator():
    time.sleep(1)
    print(x_train.shape)
    print(x_test.shape)



