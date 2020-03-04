import pandas as pd
import numpy as np
from lib.features import EloRating
from lib.utils import calc_aver
import re
from datetime import datetime


def get_cul_stats(key, raw_df):
    name_key = '%s_name' % key
    cul_wintimes_key = '%s_cul_wintimes' % key
    cul_racetimes_key = '%s_cul_racetimes' % key
    cul_winrate_key = '%s_cul_winrate' % key

    raw_df[cul_wintimes_key] = 0
    raw_df[cul_racetimes_key] = 0
    raw_df[cul_winrate_key] = 0
    uniques = raw_df[name_key].unique()
    for uni_idx, uni in enumerate(uniques):
        print('\r %s: %d/%d' % (key, uni_idx, len(uniques)), end='', flush=True)
        unique_df = raw_df[raw_df[name_key] == uni]

        unique_df_sorted = unique_df.sort_values('race_date', ascending=True)

        # Cumulative win times
        cul_wintimes = np.cumsum(unique_df_sorted['place'] == 1)
        raw_df.loc[cul_wintimes.index, cul_wintimes_key] = list(cul_wintimes)

        # Cumulative race times
        raw_df.loc[unique_df_sorted.index, cul_racetimes_key] = list(np.arange(unique_df_sorted.shape[0]))

        # Same statistics for different race numbers within a race date

        unique_df2 = raw_df[raw_df[name_key] == uni]  # Since raw_df was updated, so we query again
        unique_race_dates = unique_df2['race_date'].unique()

        for unique_race_date in unique_race_dates:
            within_racedate_df = unique_df2[unique_df2['race_date'] == unique_race_date].loc[:,
                                 ['place', cul_wintimes_key, cul_racetimes_key]]
            within_racedate_index = within_racedate_df.index

            num_wins = np.sum(within_racedate_df['place'] == 1)
            num_cul_wins = max(within_racedate_df[cul_wintimes_key]) - num_wins
            num_races = min(within_racedate_df[cul_racetimes_key])

            raw_df.loc[within_racedate_index, cul_wintimes_key] = num_cul_wins
            raw_df.loc[within_racedate_index, cul_racetimes_key] = num_races

        # # Debug
        # debug_df = raw_df[raw_df[name_key] == uni][[name_key, 'race_date', 'race_num', 'place', cul_wintimes_key, cul_racetimes_key ]].sort_values('race_date')
        # print('--'*50)
        # print(debug_df.to_string())
        # input('continue')

    wintimes = raw_df[cul_wintimes_key]
    racetimes = raw_df[cul_racetimes_key]
    raw_df[cul_winrate_key] = np.divide(wintimes, racetimes, out=np.zeros_like(wintimes, dtype=float),
                                        where=racetimes != 0)
    return raw_df


def check_nan_hythen(df):
    for col in df.columns:
        nan_counts = df[col].isna().sum()
        hythen_counts = np.sum(df[col] == '---')
        print('%s: nan = %d, "---" = %d' % (col, nan_counts, hythen_counts))

    return None


def time2sec(time_obj):
    sec = (time_obj.hour * 60 + time_obj.minute) * 60 + time_obj.second + time_obj.microsecond * 1e-6
    return sec


# Index(['place', 'horse_link' (deleted) , 'horse_no' (deleted), 'horse_code', 'horse_name', 'horse_origin',
#        'jockey_name', 'trainer_name', 'act_weight', 'decla_weight', 'draw',
#        'lbw' (deleted), 'time', 'odds', 'race_no' (deleted), 'race_num', 'track_length', 'going',
#        'course', 'race_date', 'location'],


# Load, pop, cleaning and convert dtype
df_path = 'data/raw_df.csv'
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

# Update age, weight difference
raw_df['age'] = np.nan
raw_df['time_since_last'] = np.nan
raw_df['weight_diff'] = np.nan

all_uni_horses = raw_df.horse_name.unique()
for i, uni_horse in enumerate(all_uni_horses):
    print('\rCalculate age %d/%d' % (i, len(all_uni_horses)), flush=True, end="")
    horse_df = raw_df[raw_df['horse_name'] == uni_horse]
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
    weights_plus = horse_weights_sorted[0:-1].reset_index(drop=True)
    weights_minus = horse_weights_sorted[1:].reset_index(drop=True)
    weightsdiff = (weights_plus.reset_index(drop=True) - weights_minus.reset_index(drop=True))
    complete_weightsdiff = list(weightsdiff) + [np.nan]
    raw_df.loc[sorted_indexes, 'weight_diff'] = complete_weightsdiff

    # print()
    # print('--' * 50)
    # debug_df = raw_df[ raw_df['horse_name'] == uni_horse]
    # print(debug_df[['race_date', 'age' , 'time_since_last', 'act_weight', 'weight_diff']].sort_values('race_date', ascending=False))

# Mean imputation
mean_time_since_last = np.nanmean(raw_df['time_since_last'])
raw_df.loc[raw_df['time_since_last'].isna(), 'time_since_last'] = mean_time_since_last
mean_weight_diff = np.nanmean(raw_df['weight_diff'])
raw_df.loc[raw_df['weight_diff'].isna(), 'weight_diff'] = mean_weight_diff

# Cumulative wining times, wining rates and race times
for key in ['horse', 'jockey', 'trainer']:
    key = 'horse'
    raw_df = get_cul_stats(key, raw_df)



# Elo rate


def elo_rate(key, raw_df, ratingdf_dict, elo_k=10, elo_d=400):
    elo = EloRating(elo_k, elo_d)

    name_key = '%s_name' % key
    elo_rating_key = '%s_rating' % key
    unique_players = np.sort(raw_df[name_key].unique())
    ratingdf_dict[key] = pd.DataFrame()
    ratingdf_dict[key][name_key] = unique_players
    ratingdf_dict[key]['label'] = ratingdf_dict[key][name_key].index

    ratingdf_dict[key][elo_rating_key] = 0

    raw_df[elo_rating_key] = 0
    all_uni_dates = np.sort(raw_df['race_date'].unique())
    for date_idx, uni_date in enumerate(all_uni_dates):
        uni_races = raw_df[raw_df['race_date'] == uni_date]['race_num'].unique()

        update_indexes = []
        update_ratings = []
        for uni_race in uni_races:
            print('\r%s Date: %s %d/%d' % (key, str(uni_date), date_idx, len(all_uni_dates)), end='', flush=True)
            game_df = raw_df[(raw_df['race_date'] == uni_date) & (raw_df['race_num'] == uni_race)][['place', name_key]]

            # Get ratings from storage. Note: right_index=True preserve the index of LEFT table!
            joined_df = pd.merge(game_df, ratingdf_dict[key], on=name_key, right_index=True)

            # Update ratings to raw_df. Updates shall be done before calculating new ratings,
            # because we want the past ratings before this race
            raw_df.loc[joined_df.index, elo_rating_key] = joined_df[elo_rating_key]

            # Calculate new rating
            places, ratings = np.asarray(joined_df['place']), np.asarray(joined_df[elo_rating_key])
            new_ratings, _ = elo.multi_elo(ratings=ratings, places=places)

            # Store for update
            update_indexes += list(joined_df.label)
            update_ratings += list(new_ratings)


        # Take average ratings of multiple players
        # Update the ratings at the storage. Note: update date by daye, but not immediately after race
        arr = np.stack([update_indexes, update_ratings]).T
        out = calc_aver(arr)
        temp_idx, temp_rat = out[:, 0], out[:, 1]
        ratingdf_dict[key].loc[temp_idx.astype(int), elo_rating_key] = temp_rat

    return raw_df, ratingdf_dict

ratingdf_dict = {}
key = 'horse'
for key in ['horse', 'jockey', 'trainer']:
    raw_df, ratingdf_dict = elo_rate(key, raw_df, ratingdf_dict, elo_k=10, elo_d=400)


# Debug for each ELO table update

# Tuning of elo rating



# Normalization

# One-hot encoding
