import pandas as pd
import os
from glob import glob


df_dir = 'data/intermediate_storage/stage2_single_race_dataframes'

all_dfs = glob(os.path.join(df_dir, '*.csv'))

df_list = [pd.read_csv(x, index_col=0) for x in all_dfs]

df_all = pd.concat(df_list, axis=0).reset_index(drop=True)

df_all.to_csv('data/intermediate_storage/stage3_features_engineering/raw_df.csv')