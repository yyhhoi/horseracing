import datetime
import numpy as np
def generate_date_list(newday, oldday):
    diff = newday-oldday
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