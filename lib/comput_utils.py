import numpy as np
import pandas as pd
from lib.utils import load_pickle


class BetCalculator:
    def __init__(self, bet_df_path):
        self.bet_df = load_pickle(bet_df_path)
        self.all_bet_types = self.bet_df['bet_type'].unique()
        self.all_strategies = ['win', 'place1', 'place2', 'place3', 'quinella', 'quinella12', 'quinella13',
                               'guinella23']
        self.strategy2bet = {'win': 'win', 'place1': 'place', 'place2': 'place', 'place3': 'place',
                             'quinella': 'quinella',
                             'quinella12': 'quinella_place', 'quinella13': 'quinella_place',
                             'guinella23': 'quinella_place'}
        self.max_len_betlist = {x: self.bet_df[self.bet_df['bet_type'] == x]['bet_dict'].apply(len).max() for x in
                                self.all_bet_types}

        pass

    def get_rates_return_vec(self, times, horse_nos, ymask, racecode):

        strategy_returns = {x: [] for x in self.all_strategies}
        strategy_rates = {x: [] for x in self.all_strategies}
        max1, max2, max3 = self.arr2place(times, ymask)
        num_races, num_horses = times.shape
        pred_horses1 = self._max2horse(max1, horse_nos, batch_size=num_races, num_horses=num_horses)
        pred_horses2 = self._max2horse(max2, horse_nos, batch_size=num_races, num_horses=num_horses)
        pred_horses3 = self._max2horse(max3, horse_nos, batch_size=num_races, num_horses=num_horses)
        code_df = pd.DataFrame(racecode, columns=['race_date', 'race_num'])
        merged_df = pd.merge(code_df, self.bet_df, on=['race_date', 'race_num'], right_index=True)
        for strategy in self.all_strategies:
            bet_type = self.strategy2bet[strategy]
            merged_df_bettype = merged_df[merged_df.bet_type == bet_type]
            if strategy == 'win':
                rate, earn = self._calc_winpalce_rate_earn(merged_df_bettype, pred_horses1)
            elif strategy == 'place1':
                rate, earn = self._calc_winpalce_rate_earn(merged_df_bettype, pred_horses1)
            elif strategy == 'place2':
                rate, earn = self._calc_winpalce_rate_earn(merged_df_bettype, pred_horses2)
            elif strategy == 'place3':
                rate, earn = self._calc_winpalce_rate_earn(merged_df_bettype, pred_horses3)
            elif strategy == 'quinella':
                rate, earn = self._calc_quinella_rate_earn(merged_df_bettype, pred_horses1, pred_horses2)
            elif strategy == 'quinella12':
                rate, earn = self._calc_quinella_rate_earn(merged_df_bettype, pred_horses1, pred_horses2,
                                                           place_mode=True)
            elif strategy == 'quinella13':
                rate, earn = self._calc_quinella_rate_earn(merged_df_bettype, pred_horses1, pred_horses3,
                                                           place_mode=True)
            elif strategy == 'guinella23':
                rate, earn = self._calc_quinella_rate_earn(merged_df_bettype, pred_horses2, pred_horses3,
                                                           place_mode=True)

            else:
                raise
            strategy_rates[strategy] = rate
            strategy_returns[strategy] = earn
        return strategy_rates, strategy_returns

    def _winplace_betdict2arr(self, x):
        horse_no = np.zeros(self.max_len_betlist['place']) * np.nan
        earn = horse_no.copy()
        if len(x) > 0:
            for i in range(len(x)):
                horse_no[i] = x[i][0][0]
                earn[i] = x[i][1] / 10

        return (horse_no, earn)

    def _calc_winpalce_rate_earn(self, merged_df_bettype, pred_horses):

        # Get horse's numbers and odds in numpy
        target_both = merged_df_bettype.bet_dict.apply(self._winplace_betdict2arr)
        target_both_np = np.array(list(target_both))
        target_horse_nos, target_earn = target_both_np[:, 0, :], target_both_np[:, 1, :]

        # Filter nans
        nonan_mask = (np.isnan(target_horse_nos[:, 0]) == False)
        pred_horses, target_horse_nos = pred_horses[nonan_mask], target_horse_nos[nonan_mask]
        target_earn = target_earn[nonan_mask]

        # Hit rate
        hit_mask = pred_horses.reshape(-1, 1) == target_horse_nos
        any_hit_mask = np.sum(hit_mask, axis=1) > 0
        mean_hit_rate = np.mean(any_hit_mask > 0)

        # Revenue
        profit_ratio = np.sum(target_earn[hit_mask])/hit_mask.shape[0]

        return mean_hit_rate, profit_ratio

    def _quinella_betdict2arr(self, x):
        horse_no1 = np.zeros(self.max_len_betlist['quinella']) * np.nan
        horse_no2 = horse_no1.copy()
        earn = horse_no1.copy()
        if len(x) > 0:
            for i in range(len(x)):
                horse_no1[i] = x[i][0][0]
                horse_no2[i] = x[i][0][1]
                earn[i] = x[i][1] / 10

        return (horse_no1, horse_no2, earn)

    def _quinella_place_betdict2arr(self, x):
        horse_no1 = np.zeros(self.max_len_betlist['quinella_place']) * np.nan
        horse_no2 = horse_no1.copy()
        earn = horse_no1.copy()
        if len(x) > 0:
            for i in range(len(x)):
                horse_no1[i] = x[i][0][0]
                horse_no2[i] = x[i][0][1]
                earn[i] = x[i][1] / 10
        return (horse_no1, horse_no2, earn)

    def _calc_quinella_rate_earn(self, merged_df_bettype, pred_horses1, pred_horses2, place_mode=False):
        if place_mode:
            target_both = merged_df_bettype.bet_dict.apply(self._quinella_place_betdict2arr)
        else:
            target_both = merged_df_bettype.bet_dict.apply(self._quinella_betdict2arr)
        target_both_np = np.array(list(target_both))
        target_horse_nos1, target_horse_nos2, target_earn = target_both_np[:, 0, :], target_both_np[:, 1,
                                                                                     :], target_both_np[:, 2, :]

        # Filter nan
        nonan_mask = (np.isnan(target_horse_nos1[:, 0]) == False)
        target_horse_nos1, target_horse_nos2 = target_horse_nos1[nonan_mask], target_horse_nos2[nonan_mask]
        pred_horses1, pred_horses2 = pred_horses1[nonan_mask], pred_horses2[nonan_mask]
        target_earn = target_earn[nonan_mask]

        # Hit rate
        hitmask11 = (pred_horses1.reshape(-1, 1) == target_horse_nos1)
        hitmask12 = (pred_horses1.reshape(-1, 1) == target_horse_nos2)
        hitmask21 = (pred_horses2.reshape(-1, 1) == target_horse_nos1)
        hitmask22 = (pred_horses2.reshape(-1, 1) == target_horse_nos2)
        hit_mask1 = hitmask11 | hitmask12
        hit_mask2 = hitmask21 | hitmask22
        hit_mask = hit_mask1 & hit_mask2
        any_hit_mask = np.sum(hit_mask, axis=1) > 0
        mean_hit_rate = np.mean(any_hit_mask > 0)

        # Revenue
        profit_ratio = np.sum(target_earn[hit_mask])/hit_mask.shape[0]
        return mean_hit_rate, profit_ratio

    @staticmethod
    def _max2horse(max_index, horse_nos, batch_size, num_horses):
        dummy_index = np.zeros((batch_size, num_horses))
        dummy_index[:] = np.arange(num_horses).reshape(1, -1)
        horse_mask = dummy_index == max_index.reshape(-1, 1)
        max_horses = horse_nos[horse_mask]
        return max_horses

    @staticmethod
    def arr2place(arr, ymask):
        """

        Args:
            arr: (m, nhorses). Place or time
            ymask: (m, nhorses), 1 = valid, 0 = invalid

        Returns:

        """
        sorted = BetCalculator.masked_sort(arr, ymask)
        first_max = sorted[:, 0]
        second_max = sorted[:, 1]
        third_max = sorted[:, 2]

        return first_max, second_max, third_max

    @staticmethod
    def masked_sort(arr, ymask):
        namask = np.invert(ymask.astype(bool))
        masked_arr = np.ma.masked_array(data=arr, mask=namask)
        sorted = masked_arr.argsort(axis=1, endwith=True)
        return sorted


if __name__ == "__main__":
    # # Test arr2place with mask
    # time = np.array([
    #     [5, 8, 2, 3, 0, 1],
    #     [3, 2, 6, 8, 7, 1],
    #     [9, 2, 3, 7, 6, 4]
    #
    # ])
    #
    # mask = np.array([
    #     [0, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1]
    # ]).astype(float)
    # ymask = 1- mask
    #
    # first_max, second_max, third_max = BetCalculator.arr2place(time, ymask)
    # print("Time\n", time, "\nsorted\n", sorted, "\nfirst_max\n", first_max, "\nsecond_max\n", second_max, "\nthird_max\n", third_max )


    # Test returns
    # race_df_pth = "data/training_dataframes/processed_df.pickle"
    #
    # bc = BetCalculator(bet_df_path="data/training_dataframes/bet_df.pickle")
    #
    # dg = ShuffledDataGenerator(df_path=race_df_pth, batch_size=512)
    #
    # place1_list = []
    # place2_list = []
    # place3_list = []
    #
    # for train, test in dg.iterator():
    #     x_train, y_train, racecode_train = train
    #
    #     x_test, y_test, racecode_test = test
    #
    #     places, times, horse_nos, masks = y_test[:, :, 1], y_test[:, :, 2], y_test[:, :, 3], y_test[:, :, 4]
    #
    #     # times = np.random.randint(0, 1000, size=times.shape)
    #
    #     print('Calculate Bets')
    #     strategy_rates, strategy_returns = bc.get_rates_return_vec(times, horse_nos, masks, racecode_test)
    #
    #     for strategy in strategy_rates.keys():
    #         print("%s : rate = %0.8f , return = %0.8f" % (
    #             strategy, strategy_rates[strategy], strategy_returns[strategy]
    #         )
    #               )
    #     print()

    pass
