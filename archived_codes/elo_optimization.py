import os
import pickle
import pandas as pd
import numpy as np
from lib.features import EloRating, calc_aver, EloEstimator,time2sec




if __name__ == "__main__":
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

    elo = EloEstimator(df = raw_df)
    K_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]
    D_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]
    # K_list = [1, 10]
    # D_list = [1, 10]
    elo.grid_sesarch(key=None, K_list=K_list, D_list=D_list)
