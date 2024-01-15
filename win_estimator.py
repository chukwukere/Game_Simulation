import numpy as np
import pandas as pd
import json
from pandas import json_normalize
from flask import jsonify
from scipy.optimize import minimize as sc_min
from scipy import stats
from xlsxwriter import workbook
from pymoo.core import problem
from pymoo.algorithms.moo.nsga2 import NSGA2
import plotly.express as px
from pymoo.algorithms.soo.nonconvex import ga
from pymoo.algorithms.soo.nonconvex import cmaes
from pymoo.optimize import minimize


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
        "user_play": [np.random.randint(low=0, high=101, size=size) for _ in range(1, play)],
        "user_Id": [generate_random_alphanumeric(5) for _ in range(1, play)],
        "stake": [np.random.randint(low=50, high=101) for _ in range(1, play)],
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


# def optimizationMethod(games,data_set):
#     """
#     :data_set: (array).
#     :games:
#     :return:
#     """
#     # Optimization Method 1
#     # In this optimization method, Scipy SOO is used to optimize the result array to get the best winratio
#     def ObjectiveFunction(arg):
#         return winratio(result=arg, game=games, data_set=data_set)["win_ratio"]
#
#     # define a constraint for the optimizer
#     def Customconstraint(arg):
#         return -winratio(result=arg, game="any_2", data_set=data_set)["win_ratio"]
#
#     contraint = ({"type": "ineq", "fun": Customconstraint})
#
#     optimizer = sc_min(ObjectiveFunction, x0=guess, constraints=contraint)
#
#     res = {"optimized_array": np.round(optimizer.x).astype(int),
#            "corresponding_winratio": optimizer.fun}
#
#     return np.array(res["optimized_array"])

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


# print(winratio(result=dopedResult(sample_generator(10, 10)), game=['exact_1','any_2'], data_set=sample_generator(10, 10)))
## Comparison data
# final_data = {
#     "Methods": ["Actual", "Actual_random", "Method_1", "Method_2", "adjusted_method_2"],
#     "Number of winners": [winratio(result=random_from_array)["number_of_wins"],
#                           winratio(result=guess)["number_of_wins"],
#                           winratio(result=res["optimized_array"])["number_of_wins"],
#                           winratio(result=leastOccuring(data_to_use))["number_of_wins"],
#                           winratio(result= dopedResult(data_to_use))["number_of_wins"]
#                           ],
#     "Win ratio": [winratio(result=random_from_array)["win_ratio"],
#                   winratio(result=guess)["win_ratio"],
#                   winratio(result=res["optimized_array"])["win_ratio"],
#                   winratio(result= leastOccuring(data_to_use))["win_ratio"],
#                   winratio(result= dopedResult(data_to_use))["win_ratio"]
#                   ]
# }
# # Arranging the data for analysis
# final_data2 = {
#     "Actual": winratio(result=random_from_array)["number_of_wins"],
#     "Actual_random": winratio(result=guess)["number_of_wins"],
#     "Method_1": winratio(result=res["optimized_array"])["number_of_wins"],
#     "Method_2": winratio(result=leastOccuring(data_to_use))["number_of_wins"],
#     "adjusted_method_2": winratio(result= dopedResult(data_to_use))["number_of_wins"]
#
# }

## ANALYSIS: Impact of the algoritms on the win rate over a range of plays.
# The point of this analysis is to identify the best performing algorithms and identify what regimes to use one
# algorithm over the other. I.e. an algorithm might give the lowest win rate, but might not be ideal in a certain
# play regime.

# generating the mode number of wins from the actual data. PS: the actual data is the base scenario
# this means that the actual method will represent a scenario were the system was randomly generating
# data for the result array.
# Actual = {
#     "no_of_plays": [],
#     "mode_value": [],
#     "mode_count": [],
#     "average_value": []
# }
#
# for i in range(50, 1001, 50):
#     runs = np.array([winratio(result=guess, game="any_5", data_set=dataset)["number_of_wins"]
#                      for dataset in mainDataGen(no_of_runs=100, play=i, size=10)])
#     Actual["no_of_plays"].append(i)
#     Actual["mode_value"].append(stats.mode(runs, keepdims=False)[0])
#     Actual["mode_count"].append(stats.mode(runs, keepdims=False)[1])
#     Actual["average_value"].append(np.round(np.average(runs), 0))
# #
# Actual_random = {
#     "no_of_plays": [],
#     "mode_value": [],
#     "mode_count": [],
#     "average_value": []
# }
# #
# for i in range(50, 1001, 50):
#     runs = np.array([winratio(result=random_from_array(dataset), game="any_5", data_set=dataset)["number_of_wins"]
#                      for dataset in mainDataGen(no_of_runs=100, play=i, size=10)])
#     Actual_random["no_of_plays"].append(i)
#     Actual_random["mode_value"].append(stats.mode(runs, keepdims=False)[0])
#     Actual_random["mode_count"].append(stats.mode(runs, keepdims=False)[1])
#     Actual_random["average_value"].append(np.round(np.average(runs), 0))
#
# Method_1 = {
#     "no_of_plays": [],
#     "mode_value": [],
#     "mode_count": [],
#     "average_value": []
# }
#
# for i in range(50, 1001, 50):
#     runs = np.array([winratio(result=optimizationMethod(games="any_2",data_set=dataset), game="any_2", data_set=dataset)["number_of_wins"]
#                      for dataset in mainDataGen(no_of_runs=100, play=i, size=10)])
#     Method_1["no_of_plays"].append(i)
#     Method_1["mode_value"].append(stats.mode(runs, keepdims=False)[0])
#     Method_1["mode_count"].append(stats.mode(runs, keepdims=False)[1])
#     Method_1["average_value"].append(np.round(np.average(runs), 0))

# Method_2 = {
#     "no_of_plays": [],
#     "mode_value": [],
#     "mode_count": [],
#     "average_value": []
# }
#
# for i in range(50, 1001, 50):
#     runs = np.array([winratio(result=leastOccuring(data_set=dataset), game="any_5", data_set=dataset)["number_of_wins"]
#                      for dataset in mainDataGen(no_of_runs=100, play=i, size=10)])
#     Method_2["no_of_plays"].append(i)
#     Method_2["mode_value"].append(stats.mode(runs, keepdims=False)[0])
#     Method_2["mode_count"].append(stats.mode(runs, keepdims=False)[1])
#     Method_2["average_value"].append(np.round(np.average(runs), 0))
# #
# adjusted_method_2 = {
#     "no_of_plays": [],
#     "mode_value": [],
#     "mode_count": [],
#     "average_value": []
# }
#
# for i in range(50, 1001, 50):
#     runs = np.array([winratio(result=dopedResult(data_set=dataset), game="any_5", data_set=dataset)["number_of_wins"]
#                      for dataset in mainDataGen(no_of_runs=100, play=i, size=10)])
#     adjusted_method_2["no_of_plays"].append(i)
#     adjusted_method_2["mode_value"].append(stats.mode(runs, keepdims=False)[0])
#     adjusted_method_2["mode_count"].append(stats.mode(runs, keepdims=False)[1])
#     adjusted_method_2["average_value"].append(np.round(np.average(runs), 0))
#
# algorithm_plot = px.line(data_frame=pd.DataFrame(Actual), x="no_of_plays", y="mode_value",
#                          labels={"x": "Number of plays", "y": "Actual"},
#                          hover_data=["mode_value", "mode_count", "average_value"],
#                          color_discrete_sequence=["blue"])
#
# # Plot for Actual_random
# algorithm_plot.add_trace(px.line(data_frame=pd.DataFrame(Actual_random), x="no_of_plays", y="mode_value",
#                                  labels={"x": "Number of plays", "y": "Actual_random"},
#                                  hover_data=["mode_value", "mode_count", "average_value"],
#                                  color_discrete_sequence=["orange"]).data[0])
# # # Plot for Method_1
# # algorithm_plot.add_trace(px.line(data_frame=pd.DataFrame(Method_1), x="no_of_plays", y="mode_value",
# #                                  labels={"x": "Number of plays", "y": "Method_1"},
# #                                  hover_data=["mode_value", "mode_count", "average_value"],
# #                                  color_discrete_sequence=["purple"]).data[0])
#
# # Plot for Method_2
# algorithm_plot.add_trace(px.line(data_frame=pd.DataFrame(Method_2), x="no_of_plays", y="mode_value",
#                                  labels={"x": "Number of plays", "y": "Method_2"},
#                                  hover_data=["mode_value", "mode_count", "average_value"],
#                                  color_discrete_sequence=["green"]).data[0])
#
# # Plot for adjusted_Method_2
# algorithm_plot.add_trace(px.line(data_frame=pd.DataFrame(adjusted_method_2), x="no_of_plays", y="mode_value",
#                                  labels={"x": "Number of plays", "y": "adjusted_Method_2"},
#                                  hover_data=["mode_value", "mode_count", "average_value"],
#                                  color_discrete_sequence=["red"]).data[0])
# # update plot labels
# algorithm_plot.update_layout(
#     title="Comparison of Algorithms",
#     xaxis_title="Number of plays",
#     yaxis_title="Number of winners"
# )
#
# final_data = {
#     "Methods": ["Actual", "Actual_random", "Method_2", "adjusted_method_2"],
#     "Number of winners": [Actual["mode_value"],
#                           Actual_random["mode_value"],
#                           # Method_1["mode_value"],
#                           Method_2["mode_value"],
#                           adjusted_method_2["mode_value"]
#                           ],
#     "Winners count": [Actual["mode_count"],
#                       Actual_random["mode_count"],
#                       # Method_1["mode_count"],
#                       Method_2["mode_count"],
#                       adjusted_method_2["mode_count"]
#                       ],
#     "Average number of winners": [Actual["average_value"],
#                                   Actual_random["average_value"],
#                                   # Method_1["average_value"],
#                                   Method_2["average_value"],
#                                   adjusted_method_2["average_value"]
#                                   ],
# }
# writing the data to excel
# excel_path = "C:/Users/ChukwukereUhuegbulem/Downloads/python data.xlsx"
#
# with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#     pd.DataFrame(final_data).to_excel(writer,sheet_name='any_5', index=False)
# print(algorithm_plot.show())

# The oddsoptimization is used to find the best odds for a given game considering the
# revenue, profit, and payouts.
# class OddsOptimization(problem.Problem):
#     def __init__(self, no_wins, no_plays, **kwargs):
#         super(OddsOptimization, self).__init__(
#             n_var=1,  # the decision variable - odds
#             n_obj=3,
#             xu=np.array([60]),
#             xl=np.array([2]),
#             elementwise_evaluation=True
#         )
#         self.no_wins = no_wins
#         self.no_plays = no_plays
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         odds = x
#         stake = 100
#         payouts = odds * self.no_wins * stake
#         revenue = self.no_plays * stake
#         profit = revenue - payouts
#         out["F"] = np.column_stack([revenue * np.ones_like(odds), payouts, profit])
#         # out["G"] = [cons1]
#
#
# Problem = OddsOptimization(no_wins=36, no_plays=525)
# algorithm = NSGA2(pop_size=100)
# res = minimize(problem=Problem, algorithm=algorithm, termination=("n_gen", 100), seed=1)
#
# pf = res.F
# ps = res.X
# data_for_plot = pd.DataFrame({
#     "revenue": pf[:, 0],
#     "payouts": pf[:, 1],
#     "odds": ps.flatten(),
#     "profit": pf[:, 2]
#
# })
#
# odds_plot = px.line(data_frame=data_for_plot, x="odds", y="profit",
#                     hover_data={"revenues": data_for_plot["revenue"], "payout": data_for_plot["payouts"], })
#
# excel_path = "C:/Users/ChukwukereUhuegbulem/Downloads/python data.xlsx"
#
# with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#     data_for_plot.to_excel(writer,sheet_name='odds_first7', index=False)
# print(odds_plot.show(),data_for_plot)


# LIVE simulation
# def liveSimulationStakes(play, size, per_profit, game):
#     """
#
#     :param play: (int) The number of plays e.g. 100
#     :param size: (int) the size of array for each play e.g 10
#     :param per_profit: (int) the percentage profit. to be used in calculating the odds e.g 50
#     :param game: the game to be played
#     :return:
#     """
#     # calculating the odds
#     odds = {
#         "exact_1": lambda per_profit: (-1.3125 * per_profit) + 131.25,
#         "any_2": lambda per_profit: (-0.061 * per_profit) + 6.1047,
#         "any_3": lambda per_profit: (-0.1944 * per_profit) + 19.444,
#         "any_4": lambda per_profit: (-1.75 * per_profit) + 175,
#         "any_5": 60,
#         "first_5": lambda per_profit: (-0.21 * per_profit) + 21,
#         "first_6": lambda per_profit: (-0.175 * per_profit) + 17.5,
#         "first_7": lambda per_profit: (-0.1458 * per_profit) + 14.583
#     }
#     test_data = sample_generator(play=play, size=size).copy()
#     test_data["expected_payouts"] = test_data["stake"] * np.round(odds[game](per_profit), 0)
#     stake_revenue = sum(test_data["stake"])
#
#     # calculating the result array
#     actual_random = random_from_array(test_data["user_play"]).copy()
#     doped_result = dopedResult(test_data["user_play"]).copy()
#     least_number = leastOccuring(test_data["user_play"]).copy()
#     actual = guess.copy()
#     # select the result array based on the game.
#     if game in ("exact_1", "any_4", "any_5"):
#         result_2 = actual_random
#     elif game == "any_2":
#         result_2 = doped_result
#     elif game == "any_3":
#         result_2 = actual
#     elif game in ("first_5", "first_6", "first_7"):
#         result_2 = least_number
#
#     # calculate the payouts based on the game
#     total_payouts = {
#         "exact_1": np.sum([test_data.iloc[i, 3] if test_data.iloc[i, 0][0] == result_2[0] else 0 for i in
#                            range(len(test_data["user_play"]))]),
#         "any_2": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 2 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "any_3": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 3 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "any_4": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 4 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "any_5": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2)) >= 5 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "first_5": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:5])) > 0 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "first_6": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:6])) > 0 else 0 for i in
#              range(len(test_data["user_play"]))]),
#         "first_7": np.sum(
#             [test_data.iloc[i, 3] if len(np.intersect1d(test_data.iloc[i, 0], result_2[0:7])) > 0 else 0 for i in
#              range(len(test_data["user_play"]))])
#     }
#
#     profit = stake_revenue - total_payouts[game]
#     percentage_profit = (profit / stake_revenue) * 100
#
#     live_data = {
#         "odds": np.round(odds[game](per_profit), 0),
#         "number of winners": winratio(result=result_2,game=game,data_set=test_data['user_play'])['number_of_wins'],
#         "revenue": "{:,}".format(stake_revenue),
#         "total payouts": "{:,}".format(np.round(total_payouts[game], 0)),
#         "profit": "{:,}".format(profit),
#         "percentage_profit": np.round(percentage_profit, 1)
#
#     }
#     return live_data


# quick = [liveSimulationStakes(50, 10, 50, "first_5")["percentage_profit"] for _ in range(0, 101)]
#
# quick_data = {
#     "mode": stats.mode(quick, keepdims=False)[0],
#     "count": stats.mode(quick, keepdims=False)[1],
#     "min": np.min(quick),
#     "max": np.max(quick)
# }
# pmf = pd.Series(quick).value_counts(normalize=True)
# print(liveSimulationStakes(50, 10, 50, ["exact_1","any_2"]), quick, quick_data,pmf)
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
    # calculating the odds
    odds = {
        "exact_1": lambda per_profit: (-1.3125 * per_profit) + 131.25,
        "any_2": lambda per_profit: (-0.061 * per_profit) + 6.1047,
        "any_3": lambda per_profit: (-0.1944 * per_profit) + 19.444,
        "any_4": lambda per_profit: (-1.75 * per_profit) + 175,
        "any_5": lambda per_profit: (-1.75 * per_profit) + 175,
        "first_5": lambda per_profit: (-0.21 * per_profit) + 21,
        "first_6": lambda per_profit: (-0.175 * per_profit) + 17.5,
        "first_7": lambda per_profit: (-0.1458 * per_profit) + 14.583
    }

    total_odds = np.sum([np.round(odds[i](per_profit), 0) for i in game])
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
    percentage_profit = (profit / total_revenue) * 100

    live_data = {
        "odds per game": {i: np.round(odds[i](per_profit), 0) for i in game},
        # "total odds": total_odds,
        "number of winners": winratio(result=result_2, game=game, data_set=test_data['user_play'])['number_of_wins'],
        "revenue per game": "{:,}".format(revenue_per_game),
        "payout per game": payout_per_game,
        "profit per game": profit_per_game,
        "total revenue": "{:,}".format(int(total_revenue)),
        "total payouts": "{:,}".format(int(total_payouts)),
        "profit": "{:,}".format(int(profit)),
        "percentage_profit": np.round(percentage_profit, 1).tolist()

    }
    return live_data
print(pd.DataFrame(test),multiGameChecker(test_data=test,result_2=leastOccuring(test),game=["exact_1","any_2","any_3","any_4"],per_profit=50))
# # test_data.loc[test_data["user_play"].apply(lambda x: np.array_equal(x,row)),"stake"]
