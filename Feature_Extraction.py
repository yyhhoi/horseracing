import pandas as pd
import numpy as np
from lib.features import time2sec, calc_cul_winrates, EloEstimator, DataEncoder, columns_order, feature_cols, \
    target_cols, linear_cols, cat_cols, metainfo_cols, DataGenerator, calc_cul_time_diff
from lib.utils import find_missing
import pickle
import os
import random


class FeatureExtractor:
    def __init__(self,
                 input_df_path='data/intermediate_storage/stage3_features_engineering/raw_df.csv',
                 permanent_params_save_dir='data/read_only/',
                 processed_df_save_path='data/training_dataframes/processed_df.pickle'):
        self.raw_df = pd.read_csv(input_df_path, index_col=0)
        self.elo_optim_records_dir = 'data/intermediate_storage/stage3_features_engineering'
        self.elo_params = dict(horse=dict(K=10, D=1),
                               jockey=dict(K=1, D=1),
                               trainer=dict(K=1, D=1))
        self.permanent_params_save_dir = permanent_params_save_dir
        self.processed_df_save_path = processed_df_save_path

    def process(self, update_feature_encoding=False):
        self._clean_and_convert()
        self.raw_df = calc_cul_time_diff(self.raw_df)
        self._calc_cul_winrates_allkeys()
        self._calc_elo()
        df_all = self._encode_and_normalize(save_encoding=update_feature_encoding)
        return df_all

    def elo_hyperparams_optimization(self):
        self._clean_and_convert()
        elo = EloEstimator(df=self.raw_df, records_dir=self.elo_optim_records_dir)
        K_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]
        D_list = [1, 10, 50, 100, 250, 500, 1000, 2500, 5000]
        elo.grid_sesarch(key=None, K_list=K_list, D_list=D_list)
        pass

    def _clean_and_convert(self):
        self.raw_df.pop('horse_link')
        self.raw_df.pop('lbw')
        self.raw_df.pop('race_no')
        self.raw_df.pop('horse_no')

        # General cleaning (nan and '---')
        for col in self.raw_df.columns:
            # Clean nan
            non_nan_mask = self.raw_df[col].isna() == False
            self.raw_df = self.raw_df[non_nan_mask]

            # Clean '---'
            hythen_mask = self.raw_df[col] == '---'
            self.raw_df = self.raw_df[hythen_mask == False]

        # Clean place
        non_place_pattern = '[A-Za-z]'
        self.raw_df = self.raw_df[self.raw_df.place.str.contains(non_place_pattern) == False]
        self.raw_df['place'] = self.raw_df['place'].astype(
            float)  # First to float, in order to parse strings like "14.0"

        # Convert datatype
        self.raw_df['act_weight'] = self.raw_df['act_weight'].astype(float)
        self.raw_df['decla_weight'] = self.raw_df['decla_weight'].astype(float)
        self.raw_df['draw'] = self.raw_df['draw'].astype(float)
        self.raw_df['odds'] = self.raw_df['odds'].astype(float)
        self.raw_df['race_num'] = self.raw_df['race_num'].astype(int)
        self.raw_df['track_length'] = self.raw_df['track_length'].astype(int)

        # Convert date and time
        self.raw_df["race_date"] = pd.to_datetime(self.raw_df['race_date'], format="%Y/%m/%d")
        self.raw_df["time"] = pd.to_datetime(self.raw_df['time'], format='%M:%S.%f').dt.time
        self.raw_df['time'] = self.raw_df['time'].apply(time2sec)

        # Clean time (Only possible after time conversion)
        self.raw_df = self.raw_df[self.raw_df['time'] > 0]

        # Debugging (use only one year data)
        # self.raw_df = self.raw_df[self.raw_df['race_date'] > pd.Timestamp(2019, 1, 1)]

    def _calc_cul_winrates_allkeys(self):
        for key in ['horse', 'jockey', 'trainer']:
            self.raw_df = calc_cul_winrates(key, self.raw_df)

    def _calc_elo(self):
        elo = EloEstimator(df=self.raw_df)
        for key in ['horse', 'jockey', 'trainer']:
            elo.set_params(K=self.elo_params[key]['K'], D=self.elo_params[key]['D'])

            _, (_, rating_df) = elo.score(key, record_result=False, update_df=True)
            rating_df.to_csv(os.path.join(self.permanent_params_save_dir, '%s_rating.csv' % key))
        self.raw_df = elo.df.copy()

    def _encode_and_normalize(self, save_encoding=False):
        normer = DataEncoder(df=self.raw_df, records_dir=self.permanent_params_save_dir)

        if save_encoding:
            normer.create_category_tables()
            lsp_df = normer.create_linear_scale_params()

        # Load data for category conversion & linear scaling
        _ = normer.load_category_tables()
        _ = normer.load_linear_scale_params()
        # encoding_df, scaled_df = normer.transform2encodings(vec=False)
        encoding_df_vec, _ = normer.transform2encodings(vec=True)

        # Delete Unnecessary columns
        pop_cols = ['horse_name', 'jockey_name', 'trainer_name', 'horse_code']
        for popcol in pop_cols:
            encoding_df_vec.pop(popcol)

        # Reorganize as (1) meta infos, (2) feature vectors and (3) labels
        encoding_df_vec = encoding_df_vec[columns_order]
        metainfo_df = encoding_df_vec[metainfo_cols]
        cat_features_df = encoding_df_vec[cat_cols].apply(np.concatenate, axis=1)
        linear_features_df = encoding_df_vec[linear_cols].apply(np.array, axis=1)
        features_df = pd.concat([cat_features_df, linear_features_df], axis=1).apply(np.concatenate, axis=1)
        target_df = encoding_df_vec[target_cols]
        df_all = pd.concat([metainfo_df, features_df, target_df], axis=1)
        df_all.columns = metainfo_cols + ['features'] + target_cols
        with open(self.processed_df_save_path, 'wb') as fh:
            pickle.dump(df_all, fh)
        return df_all


if __name__ == "__main__":

    fe = FeatureExtractor(input_df_path='data/intermediate_storage/stage3_features_engineering/raw_df.csv',
                          permanent_params_save_dir='data/read_only/',
                          processed_df_save_path='data/training_dataframes/processed_df.pickle')
    df_all = fe.process(update_feature_encoding=True)
    # fe.elo_hyperparams_optimization()
