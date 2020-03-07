import numpy as np
import pandas as pd
import os
import pickle
import random
from lib.utils import calc_aver, OneHotTransformer, find_missing



class EloRating:
    def __init__(self, k, d):
        self.k = k
        self.d = d

    def multi_elo(self, ratings, places):
        """

        Args:
            ratings (numpy.darray: 1D array of current ratings for each player
            places (numpy.darray): 1D array of places. place value starting from 1, not 0

        Returns:
            updated_ratings:
            (expected, scores):

        """

        num_players = len(ratings)
        num_games = (num_players * (num_players - 1)) / 2  # C(N, 2)
        expected = np.zeros_like(ratings)

        for i in range(num_players):
            for j in range(num_players):
                if i == j:
                    continue
                else:
                    expected[i] += self.expect(ratings[i], ratings[j], self.d)
        expected = expected / num_games
        scores = (num_players - places) / num_games
        updated_ratings = ratings + self.k * (scores - expected)

        return updated_ratings, (expected, scores)

    @staticmethod
    def expect(r1, r2, d):
        return 1 / (1 + np.power(10, (r2 - r1) / d))


class EloEstimator:
    def __init__(self, df, records_dir='data/intermediate_storage/stage3_features_engineering'):
        self.K = None
        self.D = None
        self.set_flag = False
        self.df = df
        self.records_dir = records_dir

    def set_params(self, K, D):
        self.K = K
        self.D = D
        self.set_flag = True

    def grid_sesarch(self, key, K_list, D_list):
        if key:
            self._loop_for_scoring(key, K_list, D_list, record=True, update_df=False)
        else:
            for key in ['horse', 'jockey', 'trainer']:
                self._loop_for_scoring(key, K_list, D_list, record=True, update_df=False)
        return

    def _loop_for_scoring(self, key, K_list, D_list, record=True, update_df=False):
        for K_each in K_list:
            for D_each in D_list:
                self.set_params(K_each, D_each)
                score, (_, _) = self.score(key, record, update_df)
                result_set = (K_each, D_each, score)
                print("Optimizing: K={}, D={}. Score={}".format(K_each, D_each, score))

    def score(self, key, record_result=True, update_df=False):
        self._check_params_set()
        df_copy = self.df.copy()
        df_copy, rating_df = self._calc_elo_rate(key, df_copy, self.K, self.D)
        pred_winrate = self._calc_winrate(df_copy, key)
        if record_result:
            self._write_records(key, pred_winrate)
        if update_df:
            self.df = df_copy
        return pred_winrate, (df_copy, rating_df)

    def _check_params_set(self):
        if self.set_flag:
            self.set_flag = False
        else:
            raise AttributeError('Must set the paramters first.')

    def _calc_winrate(self, df, key):
        elo_key = "%s_rating" % key
        related_cols = ['race_date', 'race_num', 'place', elo_key]
        agg_dict = {'place': 'idxmin', elo_key: 'idxmax'}
        grouped_df = df[related_cols].groupby(by=['race_date', 'race_num']).agg(agg_dict)
        pred_winrate = (grouped_df['place'] == grouped_df['%s_rating' % key]).mean()
        return pred_winrate

    def _write_records(self, key, pred_winrate):
        record_path = os.path.join(self.records_dir, 'elo_params_records.txt')
        if os.path.isfile(record_path):
            with open(os.path.join(self.records_dir, 'elo_params_records.txt'), 'a') as fh:
                fh.write('%s, %0.5f, %0.5f, %0.2f\n' % (key, self.K, self.D, pred_winrate))
        else:
            with open(os.path.join(self.records_dir, 'elo_params_records.txt'), 'a') as fh:
                fh.write('key, K, D, winrate\n')
                fh.write('%s, %0.5f, %0.5f, %0.2f\n' % (key, self.K, self.D, pred_winrate))

    @staticmethod
    def _calc_elo_rate(key, df, K, D):
        """
        Example:
        ratingdf_dict = {}
        for key in ['horse', 'jockey', 'trainer']:
            raw_df, ratingdf_dict = elo_rate(key, raw_df, ratingdf_dict, elo_k=10, elo_d=400)

        Args:
            key (str): "horse", "jockey" or "trainer"
            raw_df (pandas.DataFrame): Dataframe contains all horse racing data. It is also returned as output for update.
            ratingdf_dict (dict): Dictionary that contains one dataframe for each key. The dataframe contains player's identity and rating information.
            elo_k (float): Parameter "K" in ELO rating.
            elo_d (float): Parmeter "D" in ELO rating.

        Returns:
            raw_df, ratingdf_dict
        """
        elo = EloRating(K, D)

        name_key = '%s_code' % key if key == 'horse' else '%s_name' % key
        elo_rating_key = '%s_rating' % key
        unique_players = np.sort(df[name_key].unique())

        rating_df = pd.DataFrame()
        rating_df[name_key] = unique_players
        rating_df['label'] = rating_df[name_key].index

        rating_df[elo_rating_key] = 0

        df[elo_rating_key] = 0
        all_uni_dates = np.sort(df['race_date'].unique())
        for date_idx, uni_date in enumerate(all_uni_dates):
            uni_races = df[df['race_date'] == uni_date]['race_num'].unique()

            update_indexes = []
            update_ratings = []
            for uni_race in uni_races:
                print('\r%s Date: %s %d/%d' % (key, str(uni_date), date_idx, len(all_uni_dates)), end='',
                      flush=True)
                game_df = df[
                    (df['race_date'] == uni_date) & (df['race_num'] == uni_race) & (df['place'] != 20)][
                    ['place', name_key]]

                # Get ratings from storage. Note: right_index=True preserve the index of LEFT table!
                joined_df = pd.merge(game_df, rating_df, on=name_key, right_index=True)

                # Update ratings to df. Updates shall be done before calculating new ratings,
                # because we want the past ratings before this race
                df.loc[joined_df.index, elo_rating_key] = joined_df[elo_rating_key]

                # Calculate new rating
                places, ratings = np.asarray(joined_df['place']), np.asarray(joined_df[elo_rating_key])
                new_ratings, _ = elo.multi_elo(ratings=ratings, places=places)

                # Store for update
                update_indexes += list(joined_df.label)
                update_ratings += list(new_ratings)

            # Take average ratings of multiple players
            # Update the ratings at the storage. Note: update date by date, but not immediately after each race
            arr = np.stack([update_indexes, update_ratings]).T
            out = calc_aver(arr)
            temp_idx, temp_rat = out[:, 0], out[:, 1]
            rating_df.loc[temp_idx.astype(int), elo_rating_key] = temp_rat
        print()
        return df, rating_df


class DataEncoder:
    def __init__(self, df, records_dir="data/read_only"):
        self.df = df
        self.records_dir = records_dir
        self.cat_dict = dict()
        self.linear_dict = dict()
        self.cat_columns = ["horse_origin", "track_length", "going", "course", "location"]
        self.linear_columns = ['act_weight', 'decla_weight', 'time', 'odds',
                               'age', 'time_since_last', 'weight_diff', 'horse_cul_wintimes',
                               'horse_cul_racetimes', 'horse_cul_winrate', 'jockey_cul_wintimes',
                               'jockey_cul_racetimes', 'jockey_cul_winrate', 'trainer_cul_wintimes',
                               'trainer_cul_racetimes', 'trainer_cul_winrate', 'horse_rating',
                               'jockey_rating', 'trainer_rating']

    def create_category_tables(self):

        for cat_col in self.cat_columns:
            save_path = os.path.join(self.records_dir, '%s_cattable.csv' % cat_col)
            uni_cat = np.sort(self.df[cat_col].unique())
            cattable = pd.DataFrame({cat_col: uni_cat, 'label': np.arange(uni_cat.shape[0]).astype(int)})
            cattable.to_csv(save_path)

    def load_category_tables(self):

        for cat_col in self.cat_columns:
            load_path = os.path.join(self.records_dir, '%s_cattable.csv' % cat_col)
            cattable = pd.read_csv(load_path, index_col=0)
            self.cat_dict[cat_col] = {cattable.loc[i, cat_col]: cattable.loc[i, 'label'] for i in
                                      range(cattable.shape[0])}

        return self.cat_dict

    def create_linear_scale_params(self):
        save_path = os.path.join(self.records_dir, 'linear_scale_params.csv')
        x_max_list, x_min_list, x_key_list = [], [], []
        for linear_col_key in self.linear_columns:
            x_max, x_min = self.df[linear_col_key].max(), self.df[linear_col_key].min()
            x_max_list.append(x_max)
            x_min_list.append(x_min)
            x_key_list.append(linear_col_key)
        lsp_df = pd.DataFrame({'col': x_key_list, 'x_max': x_max_list, 'x_min': x_min_list})
        lsp_df.to_csv(save_path)
        return lsp_df

    def load_linear_scale_params(self):
        load_path = os.path.join(self.records_dir, 'linear_scale_params.csv')
        lsp_df = pd.read_csv(load_path, index_col=0)
        self.linear_dict = {
            lsp_df.loc[i, 'col']: (lsp_df.loc[i, 'x_max'], lsp_df.loc[i, 'x_min']) for i in
            range(lsp_df.shape[0])
        }

        return self.linear_dict

    def transform2encodings(self, vec=True):
        out_df = self.df.copy()
        out_df = self._place_draw_transform(out_df)
        scaled_df = self._linear_transform(out_df)
        out_df = self._onehot_transform(scaled_df, vec)
        return out_df, scaled_df

    def scale_single(self, x, key):
        x_max, x_min = self.linear_dict[key]
        return self.scale_range(x, x_max, x_min)

    def unscale_single(self, x, key):
        x_max, x_min = self.linear_dict[key]
        return self.unscale_range(x, x_max, x_min)

    def _place_draw_transform(self, df):
        df['place'] = df['place'] - 1
        df['draw'] = df['draw'] - 1
        return df

    def _linear_transform(self, df):
        for col in self.linear_columns:
            df[col] = self.scale_range(df[col], self.linear_dict[col][0], self.linear_dict[col][1])
        return df

    def _onehot_transform(self, df, vec=True):
        for col in self.cat_columns:
            df[col] = df[col].apply(self.cat_dict[col].get)
            if vec:
                trans = OneHotTransformer(len(self.cat_dict[col].keys()))
                df[col] = df[col].apply(trans.transform)

        return df

    @staticmethod
    def scale_range(x, max_x, min_x):
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def unscale_range(scaled_x, max_x, min_x):
        return (scaled_x * (max_x - min_x)) + min_x

class DataGenerator:
    def __init__(self, df_path, m=512):
        with open(df_path, 'rb') as fh:
            self.df = pickle.load(fh)
        self.m = m
        self.df_train, self.df_test = self._sep_train_test()  # Default: 2018-2020 for test
        self.max_nhorses = 15

    def iterator(self):
        train_groups = [x for _, x in self.df_train.groupby(['race_date', 'race_num'])]
        test_groups = [x for _, x in self.df_test.groupby(['race_date', 'race_num'])]

        while True:
            random.shuffle(train_groups)
            random.shuffle(test_groups)
            x_train, y_train = self._stack_groups(train_groups)
            x_test, y_test = self._stack_groups(test_groups)
            yield (x_train, y_train), (x_test, y_test)



    def _sep_train_test(self):

        df_train = self.df[self.df['race_date'] < pd.Timestamp(2018, 1, 1)]
        df_test = self.df[self.df['race_date'] >= pd.Timestamp(2018, 1, 1)]
        return df_train, df_test

    def _stack_groups(self, df_groups):
        x_list = []
        y_list = []
        for i in range(self.m):
            df_each = df_groups[i]

            x_each = np.zeros((self.max_nhorses , 78))
            y_each = np.zeros((self.max_nhorses , 4))
            ymask_each = np.ones(self.max_nhorses )
            draw_vec = np.asarray(df_each['draw']).astype(int)
            ymask_each[find_missing(draw_vec, self.max_nhorses)] = 0
            x_each[draw_vec, :] = np.stack(df_each['features'])
            y_each[draw_vec, 0:3] = np.asarray(df_each[target_cols])
            y_each[:, 3] = ymask_each

            x_list.append(x_each)
            y_list.append(y_each)
        x_np = np.stack(x_list)  # (m, nhorses, 78)
        y_np = np.stack(y_list)  # (m, nhorses, 4)
        y_odds, y_palce, y_time, y_mask = y_np[:, :, 0], y_np[:, :, 1], y_np[:, :, 2], y_np[:, :, 3]
        return x_np, (y_odds, y_palce, y_time, y_mask)


def get_cul_stats(key, raw_df):
    name_key = '%s_code' % key if key == 'horse' else '%s_name' % key
    cul_wintimes_key = '%s_cul_wintimes' % key
    cul_racetimes_key = '%s_cul_racetimes' % key
    cul_winrate_key = '%s_cul_winrate' % key
    raw_df[cul_wintimes_key] = 0
    raw_df[cul_racetimes_key] = 0
    raw_df[cul_winrate_key] = 0
    related_cols = [name_key, 'race_date', 'place', cul_wintimes_key, cul_racetimes_key, cul_winrate_key]
    smaller_df = raw_df[related_cols].sort_values(by=[name_key, 'race_date'])

    idx, num_rows = 1, smaller_df.groupby(by=name_key).count().shape[0]
    for name, group in smaller_df.groupby(by=name_key):
        print("\r %s: %d/%d" % (key, idx, num_rows), end="", flush=True)
        cul_wintimes = np.cumsum(group['place'] == 1)
        raw_df.loc[cul_wintimes.index, cul_wintimes_key] = list(cul_wintimes)
        raw_df.loc[group.index, cul_racetimes_key] = list(np.arange(group.shape[0]))
        idx += 1
    print()

    smaller_df = raw_df[related_cols].sort_values(by=[name_key, 'race_date'])
    idx, num_rows = 1, smaller_df.groupby(by=[name_key, 'race_date']).count().shape[0]
    for name, group in smaller_df.groupby(by=[name_key, 'race_date']):
        print("\r %s: %d/%d" % (key, idx, num_rows), end="", flush=True)
        num_wins = (group['place'] == 1).sum()
        num_cul_wins = max(group[cul_wintimes_key]) - num_wins
        num_races = min(group[cul_racetimes_key])
        raw_df.loc[group.index, cul_wintimes_key] = num_cul_wins
        raw_df.loc[group.index, cul_racetimes_key] = num_races
        idx += 1
    print()

    wintimes = raw_df[cul_wintimes_key]
    racetimes = raw_df[cul_racetimes_key]
    raw_df[cul_winrate_key] = np.divide(wintimes, racetimes, out=np.zeros_like(wintimes, dtype=float),
                                        where=racetimes != 0)
    return raw_df


def time2sec(time_obj):
    sec = (time_obj.hour * 60 + time_obj.minute) * 60 + time_obj.second + time_obj.microsecond * 1e-6
    return sec


columns_order = [
    'race_date', 'race_num', 'draw', 'location', 'going', 'course', 'track_length', 'horse_origin', 'age',
    'act_weight', 'decla_weight', 'time_since_last', 'weight_diff', 'horse_cul_wintimes', 'horse_cul_racetimes',
    'horse_cul_winrate', 'jockey_cul_wintimes', 'jockey_cul_racetimes', 'jockey_cul_winrate', 'trainer_cul_wintimes',
    'trainer_cul_racetimes', 'trainer_cul_winrate', 'horse_rating', 'jockey_rating', 'trainer_rating',
    'odds', 'place', 'time'
]

feature_cols = [
    'location', 'going', 'course', 'track_length', 'horse_origin', 'age',
    'act_weight', 'decla_weight', 'time_since_last', 'weight_diff', 'horse_cul_wintimes', 'horse_cul_racetimes',
    'horse_cul_winrate', 'jockey_cul_wintimes', 'jockey_cul_racetimes', 'jockey_cul_winrate', 'trainer_cul_wintimes',
    'trainer_cul_racetimes', 'trainer_cul_winrate', 'horse_rating', 'jockey_rating', 'trainer_rating'
]

cat_cols = ['location', 'going', 'course', 'track_length', 'horse_origin']

linear_cols = ['age',
               'act_weight', 'decla_weight', 'time_since_last', 'weight_diff', 'horse_cul_wintimes',
               'horse_cul_racetimes',
               'horse_cul_winrate', 'jockey_cul_wintimes', 'jockey_cul_racetimes', 'jockey_cul_winrate',
               'trainer_cul_wintimes',
               'trainer_cul_racetimes', 'trainer_cul_winrate', 'horse_rating', 'jockey_rating', 'trainer_rating'
               ]

target_cols = [
    'odds', 'place', 'time'
]

metainfo_cols = ['race_date', 'race_num', 'draw']
