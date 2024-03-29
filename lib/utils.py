import datetime
import numpy as np
import pickle

def write_pickle(df, pth):
    with open(pth, 'wb') as fh:
        pickle.dump(df, fh)

def load_pickle(pth):
    with open(pth, 'rb') as fh:
        return pickle.load(fh)

def generate_date_list(newday, oldday):
    diff = newday - oldday
    date_list = []
    for i in range(diff.days):
        subtractor = datetime.timedelta(days=i + 1)
        new_date = newday - subtractor
        new_date_str = new_date.strftime("%Y/%m/%d")
        date_list.append(new_date_str)
    return date_list


def calc_aver(arr):
    '''
    Average values on axis 1 over duplicated indexes on axis 0
    Args:
        arr (numpy.darray): 2D array, with axis 0 as the indexes, and axis 1 as the values.

    Returns:
        out (numpy.darray): 2D array, axis 0 as the unique indexes, axis 2 as the averaged values
    '''
    unqa, ID, counts = np.unique(arr[:, 0], return_inverse=True, return_counts=True)
    out = np.column_stack((unqa, np.bincount(ID, arr[:, 1]) / counts))
    return out


class OneHotTransformer:
    def __init__(self, max_num):
        self.arr_temp = np.zeros(max_num)

    def transform(self, label):
        out = self.arr_temp.copy()
        out[label] = 1
        return out

def find_missing(lst, max_n):
    return sorted(set(range(0, max_n)) - set(lst))