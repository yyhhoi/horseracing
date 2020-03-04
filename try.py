import numpy as np
from lib.features import EloRating





if __name__ == "__main__":


    arr = np.array([[0, 1], [0, 2], [1, 3], [1, 3], [1, 4], [2, 3]])
    unqa, ID, counts = np.unique(arr[:, 0], return_inverse=True, return_counts=True)
    out = np.column_stack((unqa, np.bincount(ID, arr[:, 1]) / counts))
    print(arr)
    print(out)
    # # Test Elo
    # import pandas as pd
    # ratings = np.zeros(12)
    # np.random.seed(10)
    # places = np.random.permutation(12)+1
    #
    #
    #
    # elo = EloRating( 10, 400)
    #
    # new_ratings, (expected, scores) = elo.multi_elo(ratings, places)
    #
    # df = pd.DataFrame({'ratings': ratings, 'places': places, 'expected':expected, 'scores':scores, 'updated r': new_ratings})
    #
    # print(df)
    # print('sum of expected values = ', np.sum(expected))
    # print('sum of scores = ', np.sum(scores))