import numpy as np



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
        num_games = (num_players * (num_players - 1))/2  # C(N, 2)
        expected = np.zeros_like(ratings)

        for i in range(num_players):
            for j in range(num_players):
                if i==j:
                    continue
                else:
                    expected[i] += self.expect(ratings[i], ratings[j], self.d)
        expected = expected/num_games
        scores = (num_players - places)/num_games
        updated_ratings = ratings + self.k * (scores - expected)

        return updated_ratings, (expected, scores)

    @staticmethod
    def expect(r1, r2, d):
        return 1 / (1 + np.power(10, (r2-r1)/d))
