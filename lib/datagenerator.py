import pickle
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from lib.features import find_missing, target_cols
from lib.utils import load_pickle


class DataGenerator:
    nfeatures = 78
    seq_length = 6
    seq_nfeatures = 107
    def __init__(self, df_path, batch_size=512, test_train_sep=None):
        self.df = load_pickle(df_path)
        self.batch_size = batch_size
        self.df_train, self.df_test = self._sep_train_test(sep_time=test_train_sep)  # Default: 2018-2020 for test

        self.train_racecode, self.train_groups, self.test_racecode, self.test_groups = self._construct_groups()
        self.max_nhorses = 15  # Hard-coded. Determined by draw

        self.x_test, self.xseq_test, self.y_test, self.racecode_test = self._stack_groups(self.test_groups,
                                                                                          racecodes=self.test_racecode,
                                                                                          start=0,
                                                                                          stop=len(
                                                                                                self.test_groups))  # stack all test races as test vector
        self.y_labels = ['odds', 'place', 'time', 'horse_no', 'mask']

    def iterator(self):
        """
        Randomly sample the indexes from data frame, and yield the sampled batch with the same indexes
        Returns
        -------
        """

        duration_indices = []
        start = 0
        for stop in range(0, len(self.train_groups), self.batch_size):
            if stop - start > 0:
                duration_indices.append((start, stop))
                start = stop

        self.train_groups, self.train_racecode = shuffle(self.train_groups, self.train_racecode)

        for start, stop in duration_indices:
            x_train, xseq_train, y_train, racecode_train = self._stack_groups(self.train_groups, self.train_racecode, start, stop)

            self._extra_operation()
            train_info = (x_train, xseq_train, y_train, racecode_train)
            test_info = (self.x_test, self.xseq_test, self.y_test, self.racecode_test)
            yield train_info, test_info

    def _sep_train_test(self, sep_time=None):
        if sep_time is None:
            sep_time = pd.Timestamp(2018, 1, 1)
        df_train = self.df[self.df['race_date'] < sep_time]  # 2018, 1, 1
        df_test = self.df[self.df['race_date'] >= sep_time]
        return df_train, df_test

    def _stack_groups(self, df_groups, racecodes, start, stop):
        x_list = []
        x_seq_list = []
        y_list = []
        racecodes_list = []
        current = start
        while current < stop:
            df_each_group = df_groups[current]
            racecodes_list.append(racecodes[current])
            x_each, x_seq_each, y_each = self._form_row_vector(df_each_group, num_rows=self.max_nhorses)
            x_list.append(x_each)
            x_seq_list.append(x_seq_each)
            y_list.append(y_each)
            current += 1

        # Stack as np array
        x_np = np.stack(x_list)  # (m, nhorses, 78)
        x_seq_np = np.stack(x_seq_list)  # (m, nhorses, 6, 107)
        y_np = np.stack(y_list)  # (m, nhorses, 4) 'odds', 'place', 'time', 'mask'

        racecodes_np = np.array(racecodes_list)  # (m, ) 'race_date', 'race_num'
        return x_np, x_seq_np, y_np, racecodes_np

    @staticmethod
    def _form_row_vector(df_each_group, num_rows):

        x_each = np.zeros((num_rows, DataGenerator.nfeatures))
        x_seq_each = np.zeros((num_rows, DataGenerator.seq_length, DataGenerator.seq_nfeatures))
        y_each = np.zeros((num_rows, 5))
        ymask_each = np.ones(num_rows)
        draw_vec = np.asarray(df_each_group['draw']).astype(int)
        ymask_each[find_missing(draw_vec, num_rows)] = 0
        x_each[draw_vec, :] = np.stack(df_each_group['features'])
        x_seq_each[draw_vec, :, :] = np.stack(df_each_group['past_records'])
        y_each[draw_vec, 0:4] = np.asarray(df_each_group[target_cols])
        y_each[:, 4] = ymask_each
        return x_each, x_seq_each, y_each  # y_each ~ 'odds', 'place', 'time', 'mask'

    def _extra_operation(self):
        pass

    def _construct_groups(self):
        train_racecode, test_racecode, train_groups, test_groups = [], [], [], []
        for racecode, df_group in self.df_train.groupby(['race_date', 'race_num']):
            train_groups.append(df_group)
            train_racecode.append(racecode)
        for racecode, df_group in self.df_test.groupby(['race_date', 'race_num']):
            test_groups.append(df_group)
            test_racecode.append(racecode)

        return train_racecode, train_groups, test_racecode, test_groups


class ShuffledDataGenerator(DataGenerator):
    """
    x ~ (m, nhorses, 79): last dimension is "draw"/num_rows
    y ~ (m, nhorses, 4): 'odds', 'place', 'time', 'mask'

    """
    nfeatures = 79

    @staticmethod
    def _form_row_vector(df_each_group, num_rows):
        x_each = np.zeros((num_rows, 79))
        y_each = np.zeros((num_rows, 5))
        ymask_each = np.ones(num_rows)

        # Assignment
        num_df_rows = df_each_group.shape[0]
        ymask_each[num_df_rows:num_rows] = 0
        x_each[0:num_df_rows, 0:78] = np.stack(df_each_group['features'])
        x_each[0:num_df_rows, 78] = np.array(df_each_group['draw'] / num_rows)
        y_each[0:num_df_rows, 0:4] = np.asarray(df_each_group[target_cols])
        y_each[:, 4] = ymask_each

        # Shuffle
        perm_vec = np.random.permutation(num_rows)
        x_each = x_each[perm_vec, :]
        y_each = y_each[perm_vec, :]

        return x_each, y_each

    def _extra_operation(self):
        self.x_test, self.y_test = self._restack_permute_arr(self.x_test, self.y_test)
        pass

    @staticmethod
    def _restack_permute_arr(x_arr, y_arr):
        # Assume x_arr and y_arr have shape (a, b, c1) and (a, b, c2). We restack and permute the dimension b
        assert x_arr.shape[0] == y_arr.shape[0]
        assert x_arr.shape[1] == y_arr.shape[1]
        dim_b = x_arr.shape[1]
        x_arr_out, y_arr_out = np.zeros_like(x_arr), np.zeros_like(y_arr)
        for dim_a in range(x_arr.shape[0]):
            perm_vec = np.random.permutation(dim_b)
            x_arr_out[dim_a, :, :] = x_arr[dim_a, perm_vec,]
            y_arr_out[dim_a, :, :] = y_arr[dim_a, perm_vec,]
        return x_arr_out, y_arr_out


if __name__ == "__main__":
    datagen = DataGenerator(df_path='/mnt/sda4/horseracing/data/training_dataframes/processed_df.pickle',
                            batch_size=512)
