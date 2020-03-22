import pandas as pd
import os
from glob import glob
from lib.utils import load_pickle, write_pickle


# combine race df
race_df_dir = 'data/intermediate_storage/stage2_single_race_dataframes'
all_race_dfs = glob(os.path.join(race_df_dir, '*.csv'))
race_df_list = [pd.read_csv(x, index_col=0) for x in all_race_dfs]
race_df_all = pd.concat(race_df_list, axis=0).reset_index(drop=True)
race_df_all.to_csv('data/intermediate_storage/stage3_features_engineering/raw_df.csv')


# combine bet df
bet_df_dir = 'data/intermediate_storage/stage2_bet_tables'
all_bet_dfs = glob(os.path.join(bet_df_dir, '*.pickle'))
bet_df_list = [load_pickle(x) for x in all_bet_dfs]
bet_df_all = pd.concat(bet_df_list, axis=0).reset_index(drop=True)
bet_df_all['race_num'] = bet_df_all['race_num'].astype(int)
bet_df_all['race_date'] = pd.to_datetime(bet_df_all['race_date'], format="%Y/%m/%d")


all_bet_types = ['win', 'place', 'quinella', 'quinella_place']
append_df_dict = {col:[] for col in bet_df_all.columns}
for name, group in bet_df_all.groupby(by=['race_date', 'race_num']):
    if group.shape[0] < 4:
        unique_types = group.bet_type.unique()
        for must_have in all_bet_types:
            if must_have not in unique_types:
                append_df_dict['race_date'].append(name[0])
                append_df_dict['race_num'].append(name[1])
                append_df_dict['bet_type'].append(must_have)
                append_df_dict['bet_dict'].append([])
append_df = pd.DataFrame(append_df_dict)
bet_df_all_complete = pd.concat([bet_df_all, append_df], axis=0).reset_index(drop=True)

write_pickle(bet_df_all_complete, 'data/training_dataframes/bet_df.pickle')


