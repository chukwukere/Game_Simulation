import numpy as np
import pandas as pd


def sample_generator(play, size):
    """

    :param play: (int) this is the number of plays. How many different instances of play
    :param size: (int) this is the size of each play. I.e. how many numbers are in the array for each play.
    :return: (dict) returns a dictionary with the play data.
    """

    # generating sample data
    def generate_random_alphanumeric(length):
        # Define the characters that can be used
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        # Use numpy's random.choice to generate a random alphanumeric string
        random_string = ''.join(np.random.choice(list(characters), size=length))

        return random_string

    data = {
        "user_play": np.array([np.random.randint(low=0, high=101, size=size) for _ in range(1, play)]).tolist(),
        "user_Id": np.array([generate_random_alphanumeric(5) for _ in range(1, play)]).tolist(),
        "stake": np.array([np.random.randint(low=50, high=101) for _ in range(1, play)]).tolist(),
    }
    return data


def mainDataGen(no_of_runs, play, size):
    """
    This function is main needed when running the analysis. It can generate sample data for the number of runs required
     in the analysis
    :param no_of_runs:
    :param play: number of plays in one run
    :param size: the size of each play.
    :return: it returns an array of sample data, where each sample can contain x amount of plays.
    """
    main_data_set = [sample_generator(play=play, size=size)["user_play"] for _ in range(0, no_of_runs)]
    return main_data_set


# this is an example data that can be used
data_to_use = mainDataGen(1, 100, 10)[0].copy()


# result from data_set
def random_from_array(data_set):
    """
    This function takes the dataset of plays and selects random numbers from
    the data set as the wining number.
    :param data_set: (dict) data that contains the plays, stakes, and user id
    :return: (list) a list of the wining number
    """
    data = pd.DataFrame(data_set)
    data = data['user_play']
    result = np.random.choice(np.concatenate(data), size=10)
    return np.array(result).tolist()


# initial guess
guess = np.random.randint(low=0, high=101, size=10)


def winratio(result, game, data_set):
    """

        :param game: (array) this informs the function on the game rule to adopt when defining a winner
        :param result: (array) this is the result array that is compared with the selection of each player
        :param data_set: (DataFrame) this is the data set that contains all the selections of each player
        :return: (dict) the function returns a dictionary that contains the win ratio and number of winners
        """
    result = np.array([int(i) for i in result])

    game_rules = {
        "exact_1": [1 if row[0] == result[0] else 0 for row in data_set],
        "any_2": [1 if len(np.intersect1d(row, result)) >= 2 else 0 for row in data_set],
        "any_3": [1 if len(np.intersect1d(row, result)) >= 3 else 0 for row in data_set],
        "any_4": [1 if len(np.intersect1d(row, result)) >= 4 else 0 for row in data_set],
        "any_5": [1 if len(np.intersect1d(row, result)) >= 5 else 0 for row in data_set],
        "first_5": [1 if len(np.intersect1d(row, result[0:5])) > 0 else 0 for row in data_set],
        "first_6": [1 if len(np.intersect1d(row, result[0:6])) > 0 else 0 for row in data_set],
        "first_7": [1 if len(np.intersect1d(row, result[0:7])) > 0 else 0 for row in data_set]
    }

    number_of_wins = {i: sum(game_rules[i]) for i in game}
    # number_of_wins = sum(game_rules[game])
    win_ratio = {i: (number_of_wins[i] / len(game_rules[i])) * 100 for i in game}

    win_data = {
        "win_ratio": win_ratio,
        "number_of_wins": number_of_wins
    }
    return win_data


def leastOccuring(data_set):
    """
    Optimization Method 2
    this method will find the 10 numbers with the least occurrence in the data set and use it for the result array.
    :param data_set: (array) this is the data set containing all the plays
    :return: returns the result array that contains the optimized result.
    """
    data_set = pd.DataFrame(data_set)
    user_play_column = data_set['user_play']
    # step 1 - get the unique numbers and the frequency of their occurrence
    unique_numbers, occurrence_in_set = np.unique(np.concatenate(user_play_column), return_counts=True)
    # get the indices that would sort the unique array to get the numbers that occur the least in ascending order
    indices_of_least_occurring = np.argsort(occurrence_in_set)[:10]
    # get the 10 least occurring numbers
    least_occurring_numbers = np.array(unique_numbers[indices_of_least_occurring])

    return least_occurring_numbers.tolist()


def dopedResult(data_set):
    """
    Optimization Method 3: also known as doped_method 2.
    this method will attempt to improve the least occurring method by doping the result array obtained from it
    with numbers
    :return: (array) returns the result array
    """
    data_set = pd.DataFrame(data_set)
    user_play_column = data_set['user_play']
    full_range = np.array(range(0, 101))
    prime_set = list(set(full_range) - set(np.concatenate(user_play_column)))
    doped_result = leastOccuring(data_set).copy()

    # ensuring there are no out of bounds edge cases
    if len(prime_set) > len(doped_result):
        filter_value = len(doped_result)
    else:
        filter_value = len(prime_set)
    # modifying the least occurring data to get the doped result
    for i in range(filter_value):
        doped_result[-(i + 1)] = prime_set[i]

    return np.array(doped_result).tolist()


test = sample_generator(1001, 10).copy()


# print(winratio(result=dopedResult(data_set=test['user_play']),game=["any_2"],data_set=test['user_play']))


def multiGameChecker(test_data, result_2, game, per_profit):
    """
    This function checks for the number of winners and possible payout from each game considering only one result array.
    :param test_data: (dict)
    :param result_2:
    :param game: an array containing the games to be played
    :param per_profit: (int) this parameter is used to calculate the odds for each game
    :return: it returns
    """
    test_data = pd.DataFrame(test_data)
    test_data["user_play"] = [[int(value) for value in row] for row in test_data["user_play"]]
    # result_2 = [int(value) for value in result_2]
    # per_profit = int(per_profit)
    # result_2 = np.array([int(i) for i in result_2]).tolist()
    # calculating the odds
    # Replace lambda functions with regular functions
    def exact_1_odds(per_profit):
        return np.round((-float(1.3125) * per_profit) + float(131.25), 0)

    def any_2_odds(per_profit):
        return np.round((-float(0.061) * per_profit) + float(6.1047), 0)

    def any_3_odds(per_profit):
        return np.round((-float(0.1944) * per_profit) + float(19.444), 0)

    def any_4_odds(per_profit):
        return np.round((-float(1.75) * per_profit) + float(175), 0)

    def any_5_odds(per_profit):
        return np.round((-float(1.75) * per_profit) + float(175), 0)

    def first_5_odds(per_profit):
        return np.round((-float(0.21) * per_profit) + float(21), 0)

    def first_6_odds(per_profit):
        return np.round((-float(0.175) * per_profit) + float(17.5), 0)

    def first_7_odds(per_profit):
        return np.round((-float(0.1458) * per_profit) + float(14.583), 0)

    # Updated odds dictionary
    odds = {
        "exact_1": exact_1_odds,
        "any_2": any_2_odds,
        "any_3": any_3_odds,
        "any_4": any_4_odds,
        "any_5": any_5_odds,
        "first_5": first_5_odds,
        "first_6": first_6_odds,
        "first_7": first_7_odds
    }

    # odds = {
    #     "exact_1": lambda per_profit: (-1.3125 * per_profit) + 131.25,
    #     "any_2": lambda per_profit: (-0.061 * per_profit) + 6.1047,
    #     "any_3": lambda per_profit: (-0.1944 * per_profit) + 19.444,
    #     "any_4": lambda per_profit: (-1.75 * per_profit) + 175,
    #     "any_5": lambda per_profit: (-1.75 * per_profit) + 175,
    #     "first_5": lambda per_profit: (-0.21 * per_profit) + 21,
    #     "first_6": lambda per_profit: (-0.175 * per_profit) + 17.5,
    #     "first_7": lambda per_profit: (-0.1458 * per_profit) + 14.583
    # }

    # total_odds = np.sum([np.round(odds[i](per_profit), 0) for i in game])
    # test_data["expected_payouts"] = test_data["stake"] * total_odds
    revenue_per_game = sum(test_data["stake"])
    total_revenue = revenue_per_game * len(game)

    # calculate the payouts based on the game
    expected_payouts = {
        "exact_1": np.sum([test_data.iloc[i, 2] * np.round(odds["exact_1"](per_profit))
                               if test_data.iloc[i, 0][0] == result_2[0] else 0 for i in
                               range(len(test_data["user_play"]))]),
        "any_2": np.sum([test_data.iloc[i, 2] * np.round(odds["any_2"](per_profit))
                             if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 2 else 0 for i in
                             range(len(test_data["user_play"]))]),
        "any_3": np.sum([test_data.iloc[i, 2] * np.round(odds["any_3"](per_profit))
                             if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 3 else 0 for i in
                             range(len(test_data["user_play"]))]),
        "any_4": np.sum([test_data.iloc[i, 2] * np.round(odds["any_4"](per_profit))
                             if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 4 else 0 for i in
                             range(len(test_data["user_play"]))]),
        "any_5": np.sum([test_data.iloc[i, 2] * np.round(odds["any_5"](per_profit))
                             if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 5 else 0 for i in
                             range(len(test_data["user_play"]))]),
        "first_5": np.sum([test_data.iloc[i, 2] * np.round(odds["first_5"](per_profit))
                               if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:5])) > 0 else 0 for i in
                               range(len(test_data["user_play"]))]),

        "first_6": np.sum([test_data.iloc[i, 2] * np.round(odds["first_6"](per_profit))
                               if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:6])) > 0 else 0 for i in
                               range(len(test_data["user_play"]))]),
        "first_7": np.sum([test_data.iloc[i, 2] * np.round(odds["first_7"](per_profit))
                               if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:7])) > 0 else 0 for i in
                               range(len(test_data["user_play"]))])

    }
    # calculating the profit per game. This based on the assumption that the players play each game on a separate
    # ticket. PS: if players play multiple games in one ticket, if one game cuts the ticket cuts.
    expected_profit_per_game = {
        "exact_1": revenue_per_game - expected_payouts["exact_1"],
        "any_2": revenue_per_game - expected_payouts["any_2"],
        "any_3": revenue_per_game - expected_payouts["any_3"],
        "any_4": revenue_per_game - expected_payouts["any_4"],
        "any_5": revenue_per_game - expected_payouts["any_5"],
        "first_5": revenue_per_game - expected_payouts["first_5"],
        "first_6": revenue_per_game - expected_payouts["first_6"],
        "first_7": revenue_per_game - expected_payouts["first_7"]
    }
    payout_per_game = {i: np.round(expected_payouts[i]) for i in game}
    profit_per_game = {i: np.round(expected_profit_per_game[i]) for i in game}
    total_payouts = sum(payout_per_game.values())
    profit = total_revenue - total_payouts
    percentage_profit = (profit / total_revenue) * int(100)

    live_data = {
        "odds per game": {i: float(np.round(odds[i](per_profit))) for i in game},
        # "total odds": total_odds,
        "number of winners": winratio(result=result_2, game=game, data_set=test_data['user_play'])['number_of_wins'],
        "revenue per game": "{:,}".format(int(revenue_per_game)),
        "payout per game": payout_per_game,
        "profit per game": profit_per_game,
        "total revenue": "{:,}".format(int(total_revenue)),
        "total payouts": "{:,}".format(int(total_payouts)),
        "profit": "{:,}".format(int(profit)),
        "percentage_profit": float(np.round(percentage_profit,2))

    }
    return live_data
# print(pd.DataFrame(test),multiGameChecker(test_data=test,result_2=leastOccuring(test),game=["exact_1","any_2","any_3","any_4"],per_profit=50))
# # test_data.loc[test_data["user_play"].apply(lambda x: np.array_equal(x,row)),"stake"]

result = {"my data" :{"result":[70, 75, 90, 16, 16, 42,  9, 68,  3, 36],
          "data": [70, 75, 90, 16, 16, 42,  9, 68,  3, 36],
          "result 2":{"result":50}},
          "name": "jason"
          }
# print(result["my data"]["result 2"]["result"])